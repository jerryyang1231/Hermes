import numpy as np
import torch
from torch import nn
from datasets import load_dataset
from torch.utils.data import Dataset
import whisper
from spec_augment import spec_augment
from utils import add_noise
from whisper.normalizers.basic import BasicTextNormalizer

# valid_set_list 包含的前11字符的ID
valid_set_list = ['-d8TlAGYFmc', '3h8m__iwuJ4', '5mPJOkoIu3k', '87omMWX-DTw',
                'E0-HOPE7_QU', 'EhqcvfaaYu8', 'gDDbnFcvWcQ', 'iy1fPQQSA6c',
                'kGbjIuzvPR8', 'MrwSzSVGiRE', 'yht8d59dCpo']

class YTTDTaigiTRSDataset(Dataset):
    def __init__(self, split, tokenizer, sample_rate, model_name, max_length, 
                spec_augment, noise_prob=0, noise_fn=None, lang='zh') -> None:
        super().__init__()
        
        if split == 'train':
            dataset = load_dataset("formospeech/yttd_taigi_trs", name='train', split='train')
            self.dataset = dataset.filter(lambda sample: sample['id'][:11] not in valid_set_list)
            print(f"train set size: {len(self.dataset)}")
        elif split == 'val':
            dataset = load_dataset("formospeech/yttd_taigi_trs", name='train', split='train')
            self.dataset = dataset.filter(lambda sample: sample['id'][:11] in valid_set_list)
            print(f"valid set size: {len(self.dataset)}")
        else:  # 'test'
            self.dataset = load_dataset("formospeech/yttd_taigi_trs", name='test', split='train')
            print(f"test set size: {len(self.dataset)}")

        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.lang = lang

        self.spec_augment = spec_augment
        self.noise_prob = noise_prob
        self.noise_fn = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else []
        self.text_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        lang = self.lang
        item = self.dataset[id]

        wav_data = item['audio']['array']
        wav_lens = len(wav_data)
        text = item['text']
        mandarin_text = item['text_mandarin']

        text = self.text_normalizer(text).replace(" ", "")
        mandarin_text = self.text_normalizer(mandarin_text).replace(" ", "")

        if np.random.rand() > self.noise_prob: 
            audio = wav_data.flatten().astype(np.float32)
        else:
            audio = add_noise(wav_data, self.noise_fn, noise_snr=0).flatten().astype(np.float32)
        
        audio_frames = len(audio.flatten()) // 160
        if self.max_length is not None:
            audio = whisper.pad_or_trim(audio.flatten(), length=self.max_length)
            
        n_mels = 80 if self.model_name != 'large-v3' else 128
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)

        if self.spec_augment:
            if self.spec_augment == "ls-double":
                mel = torch.from_numpy(spec_augment(mel.T.numpy(), audio_frames)).T
            elif self.spec_augment == "ls-basic":
                mel = torch.from_numpy(spec_augment(mel.T.numpy(), audio_frames, n_freq_mask=1, n_time_mask=1)).T
            else:
                raise NotImplementedError 

        dec_input_ids = [self.tokenizer.sot, 
                        self.tokenizer.special_tokens["<|{}|>".format(lang)],
                        self.tokenizer.transcribe, 
                        self.tokenizer.no_timestamps] + \
                        self.tokenizer.encode(" " + text)
        labels = dec_input_ids[1:] + [self.tokenizer.eot]

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "translations": mandarin_text,
            "wav_lens": wav_lens
        }