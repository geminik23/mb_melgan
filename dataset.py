import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from config import Config
import random



class WavMelPairDataset(Dataset):
    def __init__(self, path, files, upsample_factor, sample_length=16000):
        """
        files (List[str]) : "xxxxx_raw.npy\t xxxxx_mel.npy"
        """
        self.path = path
        self.files = files
        self.upsample_factor = upsample_factor
        self.sample_length = sample_length
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        raw, mel = self.files[index].split('\t')
        raw, mel = torch.tensor(np.load( os.path.join(self.path, raw.strip()))), torch.tensor(np.load(os.path.join(self.path, mel.strip())))

        mel_length = self.sample_length // self.upsample_factor

        # two cases
        if raw.size(-1) >= self.sample_length: # randomly select position 
            bidx = random.randint(0, len(raw) - self.sample_length)
            bidx = bidx - (bidx%self.upsample_factor)
            raw = raw[bidx:bidx+self.sample_length]

            bidx = bidx // self.upsample_factor
            mel = mel[...,bidx:bidx + mel_length]
        else:
            raw = F.pad(raw, (0, self.sample_length - raw.size(-1)))
            mel = F.pad(raw, (0, mel_length - mel.size(-1)), mode='replicate')
        return mel, raw.unsqueeze(0)


def load_ljspeech_dataset(config:Config):
    """
    Returns:
        tuple: (train dataset, test dataset)
    """
    meta = os.path.join(os.path.join(config.dataset_path, config.lj_processed_path),'meta.txt')
    
    with open(meta, 'r') as f:
        files = f.readlines()

    test_len = int(len(files)*config.val_ratio)
    train, test = files[:-test_len], files[-test_len:]
    return WavMelPairDataset(os.path.join(config.dataset_path, config.lj_processed_path), train, config.upsample_factor, config.sample_length), WavMelPairDataset(config.lj_processed_path, test, config.upsample_factor, config.sample_length)
