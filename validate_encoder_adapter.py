import os
import sys
import yaml
import types
import numpy as np
import torch
from torch import nn
from datasets import load_dataset
from torch.utils.data import Dataset
import whisper
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from tqdm import tqdm
from spec_augment import spec_augment
from utils import (
    add_noise,
    whisper_flamingo_collator,
    whisper_optimizer,
    whisper_flamingo_optimizer,
    whisper_adapter_optimizer,
    setup_logging_and_checkpoint,
    wer_cer,
    DistributedSamplerWrapper,
)
from utils_batch_samplers import SortedBatchSampler
import wandb
from pytorch_lightning.loggers import WandbLogger
from whisper.normalizers.basic import BasicTextNormalizer
from transformers import BertModel, BertTokenizer
import copy
import torch.nn.functional as F
from dataset import YTTDTaigiTRSDataset
os.environ["WANDB_MODE"] = "disabled"
os.environ['WANDB_DIR'] = 'wandb/'

"""
CUDA_VISIBLE_DEVICES=0 python -u validate_encoder_adapter.py config/audio-text/validate_encoder_adapter.yaml
"""

SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)

class DistillWhisperModule(LightningModule):
    def __init__(self, cfg, model_name, lang) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_name = model_name
        self.lang = lang

        # --- 步驟 1: 建立 Student 模型和 Encoder Adapter 的「空殼」 ---
        print("Loading student (vanilla) model")
        self.student = whisper.load_model(model_name,
                                        device='cpu',
                                        download_root='models',
                                        dropout_rate=cfg.dropout_rate,
                                        add_gated_x_attn=0,
                                        num_langs = cfg.num_langs,
                                        )

        # 假設 Whisper 的 hidden size (n_audio_state)
        whisper_hidden_size = self.student.dims.n_audio_state
        
        # 定義 Transformer Block 層
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=whisper_hidden_size,
            nhead=12,
            dim_feedforward=3072,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        # 疊加 Block
        self.encoder_adapter = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # --- 步驟 2: 載入 Checkpoint 並將權重分類載入 ---
        if cfg.student_ckpt:
            print(f"Loading weights for validation from: {cfg.student_ckpt}")
            checkpoint_root = 'models/checkpoints/'
            ckpt_path = os.path.join(checkpoint_root, cfg.student_ckpt)
            
            # 2a. 載入 PyTorch Lightning checkpoint 的 state_dict
            ckpt_state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']

            # 2b. 建立兩個新的 state_dict，分別存放 student 和 adapter 的權重
            student_sd = {}
            adapter_sd = {}

            print("Separating weights for student model and encoder_adapter...")
            # 2c. 遍歷 checkpoint 中的每一個參數，進行分類
            for key, value in ckpt_state_dict.items():
                if key.startswith("student."):
                    # 移除 "student." 前綴
                    new_key = key[len("student."):]
                    student_sd[new_key] = value
                elif key.startswith("encoder_adapter."):
                    # 移除 "encoder_adapter." 前綴
                    new_key = key[len("encoder_adapter."):]
                    adapter_sd[new_key] = value
            
            # 2d. 將分類好的權重分別載入到對應的模型中
            # 因為 student 在訓練時是凍結的，所以 student_sd 應該是空的，但載入是好習慣
            if student_sd:
                self.student.load_state_dict(student_sd, strict=False)
                print("Loaded frozen student weights.")
            
            if adapter_sd:
                self.encoder_adapter.load_state_dict(adapter_sd)
                print("Loaded trained encoder_adapter weights.")

        # --- 步驟 3: 設定為評估模式 ---
        # Teacher 在驗證時不需要，可以省略以節省記憶體
        self.teacher = None 
        
        self.student.eval()
        self.encoder_adapter.eval()
        for p in self.student.parameters():
            p.requires_grad = False
        for p in self.encoder_adapter.parameters():
            p.requires_grad = False

        # tokenizer, normalizer, bert (teacher uses bert to get xt)
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language=lang, task='transcribe')
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)
        self.special_token_set = set(self.tokenizer.special_tokens.values())

        # BERT (teacher side) — freeze bert, we only use it to produce xt
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert_model = BertModel.from_pretrained('bert-base-chinese')
        self.bert_model.eval()
        for p in self.bert_model.parameters():
            p.requires_grad = False

        # losses & hyperparams
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')  # expects log-probs input

        # kd hyperparameters (from cfg or default)
        self.kd_alpha = getattr(cfg, 'kd_alpha', 1.0)   # weight for CE_student
        self.kd_beta  = getattr(cfg, 'kd_beta', 0.5)    # weight for KL distillation
        self.kd_gamma = getattr(cfg, 'kd_gamma', 0.5)   # weight for logits MSE
        self.kd_temp  = getattr(cfg, 'kd_temp', 2.0)    # temperature for KD

        # log config for debugging
        print("KD config:", {"alpha": self.kd_alpha, "beta": self.kd_beta, "gamma": self.kd_gamma, "temp": self.kd_temp})
        # Note: we intentionally do NOT move modules to device here; Lightning will handle it.

    def forward(self, x):
        # keep simple - forward is not used for distill logic
        return self.student(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]          # mel [B, n_mels, T]
        labels = batch["labels"].long()         # [B, L]
        dec_input_ids = batch["dec_input_ids"].long()  # [B, L_dec]
        translations = batch["translations"]    # list[str]
        device = input_ids.device

        # 1) teacher: get xt from BERT (no grad)
        bert_inputs = self.bert_tokenizer(translations,
                                        return_tensors='pt',
                                        padding=True,
                                        truncation=True,
                                        max_length=448,
                                        ).to(device)
        with torch.no_grad():
            bert_outputs = self.bert_model(**bert_inputs)
            xt = bert_outputs.last_hidden_state  # [B, seq_len, hidden_size]

        # 2) encoder features (teacher & student)
        with torch.no_grad():
            audio_feat_teacher = self.teacher.encoder(input_ids)
            # teacher decoder forward (no grad)
            _, teacher_final, teacher_logits = self.teacher.decoder(dec_input_ids, audio_feat_teacher, xt_list=[xt], return_hidden=True)
            # teacher_logits: [B, T_dec, V]
        # student forward (trainable)
        audio_feat_student = self.student.encoder(input_ids)
        _, student_final, student_logits = self.student.decoder(dec_input_ids, audio_feat_student, return_hidden=True)  # [B, T_dec, V]

        V = student_logits.size(-1)

        # CE loss on student (use standard cross entropy)
        ce = self.ce_loss(student_logits.view(-1, V), labels.view(-1))

        # prepare flattened masked selections where labels != -100
        mask = (labels.view(-1) != -100)
        s_flat = student_logits.view(-1, V)[mask]  # [Nkept, V]
        t_flat = teacher_logits.view(-1, V).detach()[mask]  # detach teacher

        # KD loss (KLDiv between softened distributions)
        tau = float(self.kd_temp)
        log_p = F.log_softmax(s_flat / tau, dim=-1)
        q = F.softmax(t_flat / tau, dim=-1)
        kd = self.kl_loss(log_p, q) * (tau ** 2)

       # hidden states (masked)
        student_final_flat = student_final.view(-1, student_final.size(-1))[mask]      # [N, D]
        teacher_final_flat = teacher_final.view(-1, teacher_final.size(-1)).detach()[mask]  # [N, D]

        # cosine similarity
        cos_sim = F.cosine_similarity(student_final_flat, teacher_final_flat, dim=-1)  # [N]
        rep = (1 - cos_sim).mean()  # L_feat

        # total loss
        loss = self.kd_alpha * ce + self.kd_beta * kd + self.kd_gamma * rep

        # logging
        self.log("train/ce", ce, on_step=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/kd", kd, on_step=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/rep", rep, on_step=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_id, dataloader_idx=None):
        # similar to your original validation but only evaluate student
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        device = input_ids.device

        # 步驟 (a): 凍結的 student encoder 提取基礎語音特徵
        audio_feat_student = self.student.encoder(input_ids)

        # 步驟 (b): 使用已訓練的 encoder_adapter 進行「增強」
        enhanced_audio_feat = self.encoder_adapter(audio_feat_student)

        # 步驟 (c): 凍結的 student decoder 接收「增強後」的特徵來生成 logits
        student_logits = self.student.decoder(dec_input_ids, enhanced_audio_feat)

        labels[labels == -100] = self.tokenizer.eot

        V = student_logits.size(-1)
        loss = self.ce_loss(student_logits.view(-1, V), labels.view(-1))

        # decoding + metrics (reuse your original logic)
        tokens = torch.argmax(student_logits, dim=2)

        # Set all decoder predictions after first eot to eot
        eot_find = (torch.where(tokens == self.tokenizer.eot, 1, 0))
        for i in range(eot_find.shape[0]):
            if torch.any(eot_find[i] == 1):
                first_eot = torch.argmax(torch.arange(eot_find.shape[1], 0, -1).to(device) * eot_find[i], dim=0, keepdim=True)
                tokens[i, torch.arange(eot_find.shape[1]).to(device) > first_eot] = self.tokenizer.eot

        mask = ~(tokens[:, 3:] == self.tokenizer.eot)
        n_correct = torch.sum(tokens[:, 3:].masked_select(mask).eq(labels[:, 3:].masked_select(mask)))
        total = torch.sum(mask)
        acc = n_correct.item() / (total.item() + 1e-6)
        acc = acc if acc < 1 else 0

        o_list, l_list = [], []
        for o, l in zip(tokens, labels):
            decoded_o = self.tokenizer.decode([t for t in o if t.item() not in self.special_token_set])
            decoded_l = self.tokenizer.decode([t for t in l if t.item() not in self.special_token_set])
            normalized_o = self.text_normalizer(decoded_o).replace(" ", "")
            normalized_l = self.text_normalizer(decoded_l).replace(" ", "")
            o_list.append(normalized_o)
            l_list.append(normalized_l)

        wer, cer = wer_cer(hypo=o_list, ref=l_list)

        log_prefix = {0: 'val', 1: 'test'}
        self.log("{}/loss_student".format(log_prefix[dataloader_idx]), loss, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/cer_student".format(log_prefix[dataloader_idx]), cer, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("{}/acc_student".format(log_prefix[dataloader_idx]), acc, on_step=False, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)

        # print sample (optional)
        for i, (hypo, ref) in enumerate(zip(o_list, l_list)):
            print("="*50)
            print("PRED: {}".format(hypo))
            print("REF:  {}".format(ref))
            if i == 1: break

        return

    def configure_optimizers(self):
        # Optimize only student parameters (and optionally other trainable parts)
        params = [p for p in self.student.parameters() if p.requires_grad]
        # you can choose to include additional modules (like a small adapter) here
        optimizer, scheduler = whisper_optimizer(self.student, self.cfg, self.t_total)
        self.optimizer, self.scheduler = optimizer, scheduler
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.t_total = self.cfg.num_train_steps

    def train_dataloader(self):
        dataset = YTTDTaigiTRSDataset('train',
                                      self.tokenizer, 
                                      SAMPLE_RATE,
                                      self.model_name,
                                      max_length=self.cfg.audio_max_length,
                                      spec_augment=self.cfg.spec_augment,
                                      noise_prob=cfg.noise_prob,
                                      lang=cfg.lang,
                                      )
        batch_sampler = SortedBatchSampler(
                    batch_size = self.cfg.batch_size,
                    shapes=[(item['wav_lens']) for item in dataset],
                    sort_in_batch='descending',
                    sort_batch='descending',
                    drop_last=True)
        if cfg.num_devices > 1:
            print("Using distributed sampler")
            batch_sampler = DistributedSamplerWrapper(batch_sampler)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=batch_sampler,
                          num_workers=self.cfg.num_worker,
                          collate_fn=whisper_flamingo_collator())

    def val_dataloader(self):
        dataset = YTTDTaigiTRSDataset('val',
                                    self.tokenizer, 
                                    SAMPLE_RATE,
                                    self.model_name,
                                    max_length=self.cfg.audio_max_length,
                                    spec_augment=False,
                                    noise_prob=0,
                                    lang=cfg.lang,
                                    )
        batch_sampler = SortedBatchSampler(
                    batch_size = self.cfg.batch_size,
                    shapes=[(item['wav_lens']) for item in dataset],
                    sort_in_batch='descending',
                    sort_batch='descending',
                    drop_last=False)
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=batch_sampler,
                          num_workers=self.cfg.num_worker,
                          collate_fn=whisper_flamingo_collator())
       
    def test_dataloader(self):
        dataset = YTTDTaigiTRSDataset('test',
                                    self.tokenizer, 
                                    SAMPLE_RATE,
                                    self.model_name,
                                    max_length=self.cfg.audio_max_length,
                                    spec_augment=False,
                                    noise_prob=0,
                                    lang=cfg.lang,
                                    )
        batch_sampler = SortedBatchSampler(batch_size = self.cfg.batch_size,
                                            shapes=[(item['wav_lens']) for item in dataset],
                                            sort_in_batch='descending',
                                            sort_batch='descending',
                                            drop_last=False
                                            )
        return torch.utils.data.DataLoader(dataset,
                          batch_sampler=batch_sampler,
                          num_workers=self.cfg.num_worker,
                          collate_fn=whisper_flamingo_collator())

if __name__ == "__main__":
    cfg_yaml = sys.argv[1]
    with open(cfg_yaml, 'r') as file:
        dct = yaml.safe_load(file)
        cfg = types.SimpleNamespace(**dct)

    print(cfg)
    print("audio max length: {}".format(cfg.audio_max_length))

    # Initialize WandB (as before)
    wandb.init(project="Hermes", config=cfg, name=cfg.train_id)

    callback_list = setup_logging_and_checkpoint(cfg.log_output_dir, 
                                                cfg.check_output_dir, 
                                                cfg.train_name, 
                                                cfg.train_id,
                                                cfg.monitor,
                                                cfg.filename
                                                )

    # instantiate distillation module (teacher loaded from transasr_ckpt)
    model = DistillWhisperModule(cfg, cfg.model_name, cfg.lang)

    # Wandb logger, trainer setup as before
    wandb_logger = WandbLogger()

    strategy = DDPStrategy(find_unused_parameters=True) if cfg.num_devices > 1 else "auto"
    trainer = Trainer(
        precision=cfg.precision,
        strategy=strategy,
        accelerator="gpu",
        max_steps=cfg.num_train_steps,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=wandb_logger,
        callbacks=callback_list,
        num_sanity_val_steps=0,
        devices=cfg.num_devices,
        val_check_interval=int(cfg.validate_every_n_batches * cfg.gradient_accumulation_steps),
        check_val_every_n_epoch=None,
        reload_dataloaders_every_n_epochs=1,
        use_distributed_sampler=False,
        sync_batchnorm=True,
    )

    resume_ckpt = f"{cfg.check_output_dir}/{cfg.train_id}/last.ckpt"
    if os.path.exists(resume_ckpt) and cfg.resume_training:
        trainer.fit(model, ckpt_path='last', val_dataloaders=[model.val_dataloader(), model.test_dataloader()])
    else:
        trainer.validate(model=model, dataloaders=[model.val_dataloader(), model.test_dataloader()])
        # trainer.fit(model, val_dataloaders=[model.val_dataloader(), model.test_dataloader()])

    wandb.finish()
