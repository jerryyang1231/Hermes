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
# os.environ["WANDB_MODE"] = "disabled"
# os.environ['WANDB_DIR'] = 'wandb/'

"""
CUDA_VISIBLE_DEVICES=2 python -u hermes_asr_taigi_basic_joint_train.py config/audio-text/hermes_asr_taigi_basic_joint_train.yaml
"""

SAMPLE_RATE = 16000
SEED = 3407
seed_everything(SEED, workers=True)

class DistillWhisperModule(LightningModule):
    def __init__(self, cfg, model_name, lang) -> None:
        super().__init__()
        self.automatic_optimization = False # <--- 加入這一行，關閉自動優化
        self.cfg = cfg
        self.model_name = model_name
        self.lang = lang

        # --- 步驟 1: 載入您已經 fine-tune 好的台語 Whisper 模型作為基礎 ---
        print("Loading base fine-tuned Whisper model...")
        base_ckpt_path = cfg.base_whisper_ckpt # 您需要在 config 中指定這個路徑

        # 建立 Teacher 和 Student 的「空殼」
        # Teacher 需要 gated_x_attn 的結構
        self.teacher = whisper.load_model(model_name,
                                        device='cpu', # avoid OOM on gpu 0 for distributed
                                        download_root='/share/nas169/jerryyang/LREC_2026/Hermes/models',
                                        dropout_rate=cfg.dropout_rate,
                                        add_gated_x_attn=cfg.add_gated_x_attn,
                                        num_langs = cfg.num_langs,
                                        )
        # Student 是標準的 Whisper
        self.student = whisper.load_model(model_name,
                                        device='cpu',
                                        download_root='/share/nas169/jerryyang/LREC_2026/Hermes/models',
                                        dropout_rate=cfg.dropout_rate,
                                        add_gated_x_attn=0,  # no gated x-attn for student
                                        num_langs = cfg.num_langs,
                                        )

        # 載入基礎權重
        state_dict = torch.load(base_ckpt_path, map_location='cpu')['state_dict']
        
        # 移除 "model." 前綴
        state_dict_updated = {k.replace('model.', ''): v for k, v in state_dict.items()}

        # 將同一份權重載入到 Teacher 和 Student 中
        # strict=False 允許 Teacher 缺少 gated_x_attn 相關權重
        self.teacher.load_state_dict(state_dict_updated, strict=False)
        self.student.load_state_dict(state_dict_updated, strict=False)
        print("Initialized both Teacher and Student from the same checkpoint.")

        # --- 步驟 2: 精準地設定 Teacher 和 Student 的凍結/解凍狀態 ---
        # 對於 Teacher: 凍結所有參數，然後只解凍 gated cross-attention 相關層
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        # unnecessary?
        x_attn_keywords = ["gated_x_attn", "attn_gate", "ff_gate"] # 根據您的 optimizer
        for n, p in self.teacher.named_parameters():
            if any(keyword in n for keyword in x_attn_keywords):
                p.requires_grad = True
                print(f"Unfreezing Teacher param: {n}")

        # 對於 Student: 凍結 Encoder，解凍 Decoder
        for p in self.student.encoder.parameters():
            p.requires_grad = False
        for p in self.student.decoder.parameters():
            p.requires_grad = True

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
        self.kd_temp  = getattr(cfg, 'kd_temp', 2.0)    # temperature for KD
        self.kd_delta = getattr(cfg, 'kd_delta', 1.0)   # weight for CE_teacher

        # log config for debugging
        print("KD config:", {"alpha": self.kd_alpha, "beta": self.kd_beta, "temp": self.kd_temp, "delta": self.kd_delta})
        # Note: we intentionally do NOT move modules to device here; Lightning will handle it.

    def forward(self, x):
        # keep simple - forward is not used for distill logic
        return self.student(x)

    def training_step(self, batch, batch_id):
        # --- 1. 取得優化器 (這是手動模式的關鍵步驟) ---
        opt_teacher, opt_student = self.optimizers()

        # --- 2. 正常計算 forward pass 和 loss (這部分和您原本的程式碼完全一樣) ---
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        translations = batch["translations"]
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
        audio_feat_teacher = self.teacher.encoder(input_ids)
        teacher_logits = self.teacher.decoder(dec_input_ids, audio_feat_teacher, xt_list=[xt])
        # teacher_logits: [B, T_dec, V]

        # student forward
        audio_feat_student = self.student.encoder(input_ids)
        student_logits = self.student.decoder(dec_input_ids, audio_feat_student)  # [B, T_dec, V]

        V = student_logits.size(-1)

        # teacher CE loss
        teacher_ce = self.ce_loss(teacher_logits.view(-1, V), labels.view(-1))

        # student CE loss
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

        # total loss
        loss = self.kd_alpha * ce + self.kd_beta * kd + self.kd_delta * teacher_ce

        # --- 3. 手動執行優化步驟 ---
        # 首先清除兩個優化器的舊梯度
        opt_teacher.zero_grad()
        opt_student.zero_grad()

        # 計算梯度 (Lightning 建議使用 self.manual_backward)
        self.manual_backward(loss)

        # 依序更新兩個優化器的權重
        opt_teacher.step()
        opt_student.step()

        # --- 4. 手動更新學習率排程器 (如果有的話) ---
        sched_teacher, sched_student = self.lr_schedulers()
        sched_teacher.step()
        sched_student.step()

        # --- 5. Logging ---
        self.log("train/ce", ce, on_step=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/kd", kd, on_step=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/teacher_ce", teacher_ce, on_step=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)

        # 在手動模式下，training_step 不需要 return 任何東西

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
        # --- 優化器 1: Teacher's Optimizer ---
        # 使用您提供的 whisper_flamingo_optimizer 邏輯，只訓練 gated_x_attn 相關層
        print("Setting up Teacher's optimizer...")
        # 假設您的 config 中有 teacher 專用的學習率
        teacher_lr = getattr(self.cfg, 'learning_rate_teacher', 1e-4) 
        
        # 建立一個臨時的 config 來傳遞學習率
        teacher_cfg = types.SimpleNamespace(**self.cfg.__dict__)
        teacher_cfg.learning_rate = teacher_lr
        
        # 使用您提供的 optimizer 函數，但只作用在 teacher 模型上
        optimizer_teacher, scheduler_teacher = whisper_flamingo_optimizer(
            self.teacher, teacher_cfg, self.t_total
        )

        # --- 優化器 2: Student's Optimizer ---
        # 使用標準的 whisper_optimizer，訓練 Student 的 Decoder
        print("Setting up Student's optimizer...")
        student_lr = getattr(self.cfg, 'learning_rate_student', 1e-4)
        
        student_cfg = types.SimpleNamespace(**self.cfg.__dict__)
        student_cfg.learning_rate = student_lr
        
        optimizer_student, scheduler_student = whisper_optimizer(
            self.student, student_cfg, self.t_total
        )

        # 按照 PyTorch Lightning 的規定，回傳兩個 list
        return [optimizer_teacher, optimizer_student], [scheduler_teacher, scheduler_student]

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
