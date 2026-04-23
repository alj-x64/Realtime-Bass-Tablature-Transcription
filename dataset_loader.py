import os
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __int__(self, csv_file, root_dir, audio_length = 4608, sr = 22050):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.audio_length = audio_length
        self.sr = sr

        self.pitch_to_index = {"Silence": 0}
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for i in range(1, 64):
            midi_val = 15 + i
            octave = (midi_val // 12) - 1
            note = notes[midi_val % 12]
            pitch_str = f"{note}{octave}"
            self.pitch_to_index[pitch_str] = i

    def __len__(self):
        return len(self.annotations)
    
    def get_audio_path(self, filename):
        for root, _, files in os.walk(self.root_dir):
            if filename in files:
                return os.path.join(root, filename)
            return None
    
    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        audio_path = self.get_audio_path(row['filename'])
