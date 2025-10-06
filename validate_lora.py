import os
import sys
import yaml
import types
import torch
from torch import nn
from torch.utils.data import Dataset
import whisper
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from tqdm import tqdm
from spec_augment import spec_augment
from utils import (
    whisper_flamingo_collator,
    whisper_optimizer,
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
os.environ["WANDB_MODE"] = "disabled"
os.environ['WANDB_DIR'] = 'wandb/'

"""
CUDA_VISIBLE_DEVICES=0 python -u validate_lora.py config/audio-text/validate_lora.yaml
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

        self.student = whisper.load_model(model_name,
                                        device='cpu',
                                        download_root='models/',
                                        dropout_rate=cfg.dropout_rate,
                                        add_gated_x_attn=0,  # no gated x-attn for student
                                        num_langs = cfg.num_langs,
                                        # add_adapter = True,
                                        # adapter_dim = 256,
                                        )

        # 將 SimpleNamespace 改為字典
        if not hasattr(self.student, "config"):
            # 使用一個字典，它天生就支援 .get() 方法
            self.student.config = {"is_encoder_decoder": True}

        # # supported leaf module types for PEFT
        # _supported_types = (nn.Linear, nn.Embedding, nn.Conv2d)

        # decoder_targets = []
        # for name, module in self.student.named_modules():
        #     # only care decoder subtree
        #     if 'decoder' not in name:
        #         continue
        #     # only leaf modules of supported types
        #     if isinstance(module, _supported_types):
        #         # filter to attention-related linears (common suffixes)
        #         last_token = name.split('.')[-1]
        #         if any(k in last_token for k in ('query', 'key', 'value', 'out')):
        #             decoder_targets.append(name)

        # decoder_targets = sorted(set(decoder_targets))

        # 定義 LoRA 設定
        print("Applying LoRA to the Student model...")
        lora_config = LoraConfig(
            r=16,  # LoRA 的秩 (rank)，可以從 8, 16, 32 開始嘗試
            lora_alpha=32,  # 縮放因子，通常設為 r 的兩倍
            # target_modules=decoder_targets,   # use full module names (precise)
            target_modules=["query", "key", "value", "out"],
            lora_dropout=0.05,
            task_type=TaskType.SEQ_2_SEQ_LM # 告訴 peft 這是一個序列到序列的模型
        )

        # 用 peft 將 student 模型包裝成 PeftModel
        self.student = get_peft_model(self.student, lora_config)

        # 打印出可訓練參數的比例，這是一個非常有用的驗證步驟！
        self.student.print_trainable_parameters()

        # load student weight
        if cfg.student_ckpt != '':
            checkpoint_root = 'models/checkpoints/'
            checkpoint = torch.load(os.path.join(checkpoint_root, cfg.student_ckpt), map_location=torch.device('cpu'))
            state_dict = checkpoint["state_dict"]
            student_only_sd = {}
            for k, v in state_dict.items():
                # 保留 student. 開頭
                if k.startswith("student."):
                    new_k = k[len("student."):]  # 去除前綴
                    student_only_sd[new_k] = v
            # print(student_only_sd.keys())
            try:
                self.student.load_state_dict(student_only_sd) 
            except BaseException as e: 
                print(str(e))
                print("Loading weights with strict=False")
                self.student.load_state_dict(student_only_sd, strict=False) 

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
        self.sce_loss = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=cfg.label_smoothing)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')  # expects log-probs input

        # kd hyperparameters (from cfg or default)
        self.alpha = cfg.alpha   # weight for KL distillation
        self.beta  = cfg.beta    # weight for feat_loss
        self.gamma = cfg.gamma   # weight for CE
        self.temp  = cfg.temp    # temperature for KD

        # log config for debugging
        print("KD config:", {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "temp": self.temp})
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
        with torch.no_grad():
            audio_feat_student = self.student.encoder(input_ids)
        student_hiddens, student_final, student_logits = self.student.decoder(dec_input_ids, audio_feat_student, return_hidden=True)  # [B, T_dec, V]

        V = student_logits.size(-1)

        # student CE loss
        sce_loss = self.sce_loss(student_logits.view(-1, V), labels.view(-1))

        # logit loss
        mask = (labels.view(-1) != -100)
        s_flat = student_logits.view(-1, V)[mask]  # [Nkept, V]
        t_flat = teacher_logits.view(-1, V).detach()[mask]  # detach teacher
        tau = float(self.temp)
        log_p = F.log_softmax(s_flat / tau, dim=-1)
        q = F.softmax(t_flat / tau, dim=-1)
        logit_loss = self.kl_loss(log_p, q) * (tau ** 2)

        # feat loss
        # 假設 teacher_final 與 student_final 的 shape 是 (B, T, D) 或 (B, D)
        # 取得 sequence-level audio feature：若為 time-series，做 GAP（mean over time）
        def pool_to_sentence(x):
            # x: either (B, T, D) or (B, D)
            if x.dim() == 3:
                return x.mean(dim=1)  # GAP -> (B, D)
            else:
                return x  # already (B, D)

        t_feat = pool_to_sentence(teacher_final)   # teacher fused feature (detach)
        s_feat = pool_to_sentence(student_final)   # student feature
        t_feat = t_feat.detach()

        # cosine loss per-sample then mean
        feat_loss = (1.0 - F.cosine_similarity(s_feat, t_feat, dim=-1)).mean()

        # total loss
        loss = self.alpha * logit_loss + self.beta * feat_loss + self.gamma * sce_loss

        # logging
        self.log("train/logit_loss", logit_loss, on_step=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/feat_loss", feat_loss, on_step=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/sce_loss", sce_loss, on_step=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/total_loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)

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

        V = student_logits.size(-1)
        loss = self.sce_loss(student_logits.view(-1, V), labels.view(-1))

        labels[labels == -100] = self.tokenizer.eot 

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
