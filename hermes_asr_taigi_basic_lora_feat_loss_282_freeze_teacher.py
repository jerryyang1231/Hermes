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
from peft import LoraConfig, get_peft_model, TaskType
# os.environ["WANDB_MODE"] = "disabled"
# os.environ['WANDB_DIR'] = 'wandb/'

"""
CUDA_VISIBLE_DEVICES=2 python -u hermes_asr_taigi_basic_lora_feat_loss_282_freeze_teacher.py config/audio-text/hermes_asr_taigi_basic_lora_feat_loss_282_freeze_teacher.yaml
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

        print("Loading teacher (TransASR) model")
        # load teacher model (with gated x-attn)
        self.teacher = whisper.load_model(model_name,
                                        device='cpu', # avoid OOM on gpu 0 for distributed
                                        download_root='/share/nas169/jerryyang/LREC_2026/Hermes/models',
                                        dropout_rate=cfg.dropout_rate,
                                        add_gated_x_attn=cfg.add_gated_x_attn,
                                        num_langs = cfg.num_langs,
                                        )

        # load TransASR ckpt weights into teacher if provided
        trans_ckpt = cfg.transasr_ckpt
        if trans_ckpt != '':
            checkpoint_root = '/share/nas169/jerryyang/LREC_2026/Hermes/models/checkpoints/'
            ckpt_path = os.path.join(checkpoint_root, cfg.transasr_ckpt) if not os.path.isabs(trans_ckpt) else trans_ckpt
            print("Loading TransASR checkpoint for teacher:", ckpt_path)
            state_dict = torch.load(ckpt_path, map_location='cpu')
            state_dict = state_dict['state_dict']
            # remove possible "model." prefix
            state_dict_updated = {}
            for k, v in state_dict.items():
                newk = k
                if k.startswith('model.'):
                    newk = k[len('model.'):]
                state_dict_updated[newk] = v
            try:
                self.teacher.load_state_dict(state_dict_updated, strict=False)
            except Exception as e:
                print("Teacher load_state_dict error:", e)
                self.teacher.load_state_dict(state_dict_updated, strict=False)

        # freeze teacher fully
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        print("Loading student (vanilla) model")
        # student: vanilla whisper decoder (no gated x-attn)
        self.student = whisper.load_model(model_name,
                                        device='cpu',
                                        download_root='/share/nas169/jerryyang/LREC_2026/Hermes/models',
                                        dropout_rate=cfg.dropout_rate,
                                        add_gated_x_attn=0,  # no gated x-attn for student
                                        num_langs = cfg.num_langs,
                                        )

        # initialize student with overlapping weights from teacher where shapes match
        print("Copying overlapping weights from teacher -> student where possible")
        teacher_state_dict = self.teacher.state_dict()
        student_state_dict = self.student.state_dict()
        loaded = 0
        for k, v in teacher_state_dict.items():
            if k in student_state_dict and student_state_dict[k].shape == v.shape:
                student_state_dict[k] = v.clone()
                loaded += 1
        self.student.load_state_dict(student_state_dict, strict=False)
        print(f"Copied {loaded} matching tensors from teacher to student (approx).")

        # freeze student encoder
        for p in self.student.encoder.parameters():
            p.requires_grad = False

        # 將 SimpleNamespace 改為字典
        if not hasattr(self.student, "config"):
            # 使用一個字典，它天生就支援 .get() 方法
            self.student.config = {"is_encoder_decoder": True}

        # 3. 定義 LoRA 設定
        print("Applying LoRA to the Student model...")
        lora_config = LoraConfig(
            r=16,  # LoRA 的秩 (rank)，可以從 8, 16, 32 開始嘗試
            lora_alpha=32,  # 縮放因子，通常設為 r 的兩倍
            target_modules=["query", "value"], # 針對 Whisper 的 Attention 層中的 query 和 value 進行注入
            lora_dropout=0.05,
            task_type=TaskType.SEQ_2_SEQ_LM # 告訴 peft 這是一個序列到序列的模型
        )

        # 4. 用 peft 將 student 模型包裝成 PeftModel
        self.student = get_peft_model(self.student, lora_config)

        # 5. 打印出可訓練參數的比例，這是一個非常有用的驗證步驟！
        self.student.print_trainable_parameters()

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
        self.kd_gamma = getattr(cfg, 'kd_gamma', 0.5)   # weight for logits cos_sim
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

        bert_inputs = self.bert_tokenizer(translations,
                                        return_tensors='pt',
                                        padding=True,
                                        truncation=True,
                                        max_length=448,
                                        ).to(device)
        with torch.no_grad():
            bert_outputs = self.bert_model(**bert_inputs)
            xt = bert_outputs.last_hidden_state  # [B, seq_len, hidden_size]

        # teacher forward
        with torch.no_grad():
            audio_feat_teacher = self.teacher.encoder(input_ids)
            teacher_hiddens, teacher_final, teacher_logits = self.teacher.decoder(dec_input_ids, audio_feat_teacher, xt_list=[xt], return_hidden=True)  # [B, T_dec, V]
        # student forward
        audio_feat_student = self.student.encoder(input_ids)
        student_hiddens, student_final, student_logits = self.student.decoder(dec_input_ids, audio_feat_student, return_hidden=True)  # [B, T_dec, V]

        V = student_logits.size(-1)

        # student CE loss
        ce = self.ce_loss(student_logits.view(-1, V), labels.view(-1))

        # KD loss
        mask = (labels.view(-1) != -100)
        s_flat = student_logits.view(-1, V)[mask]  # [Nkept, V]
        t_flat = teacher_logits.view(-1, V).detach()[mask]  # detach teacher
        tau = float(self.kd_temp)
        log_p = F.log_softmax(s_flat / tau, dim=-1)
        q = F.softmax(t_flat / tau, dim=-1)
        kd = self.kl_loss(log_p, q) * (tau ** 2)

        # feat loss
        D = student_final.size(-1)
        student_final_masked = student_final.view(-1, D)[mask]      # [N, D]
        teacher_final_masked = teacher_final.view(-1, D).detach()[mask]  # [N, D]
        cos_sim = F.cosine_similarity(student_final_masked, teacher_final_masked, dim=-1)
        feat_loss = (1 - cos_sim).mean()

        # total loss
        loss = self.kd_alpha * ce + self.kd_beta * kd + self.kd_gamma * feat_loss

        # logging
        self.log("train/ce", ce, on_step=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/kd", kd, on_step=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/feat_loss", feat_loss, on_step=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_id, dataloader_idx=None):
        # similar to your original validation but only evaluate student
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        device = input_ids.device

        # student forward
        audio_feat_student = self.student.encoder(input_ids)
        student_logits = self.student.decoder(dec_input_ids, audio_feat_student)

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
        trainer.fit(model, val_dataloaders=[model.val_dataloader(), model.test_dataloader()])

    wandb.finish()
