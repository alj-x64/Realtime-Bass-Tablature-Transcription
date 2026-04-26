import os
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, csv_file, root_dir, audio_length = 4608, sr = 22050):
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
    
    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        audio_path = os.path.join(self.root_dir, row['filepath'])

        if audio_path is None:
            return torch.zeros(self.audio_length), self._get_dummy_labels()

        try:
            onset_time = float(row['onset'])
            offset_time = float(row['offset'])

            y_main, _ = librosa.load(audio_path, sr=self.sr)

            start_sample = max(0, int(onset_time * self.sr))
            end_sample = min(len(y_main), int(offset_time * self.sr))

            y_main = y_main[start_sample:end_sample]

            if len(y_main) < self.audio_length:
                y_main = np.pad(y_main, (0, self.audio_length - len(y_main)))
            else:
                y_main = y_main[:self.audio_length]

            is_active = True
            onset_label = 1.0
            offset_label = 1.0

        except Exception as e:
            y_main = np.zeros(self.audio_length, dtype=np.float32)
            is_active = False
            onset_label = 0.0
            offset_label = 0.0
        
        if is_active:
            string_label = int(row['string'])
            fret_label = int(row['fret']) + 1
            pitch_label = self.pitch_to_index.get(row['pitch'], 0)
        else:
            string_label = 0
            fret_label = 0
            pitch_label = 0
        
        audio_tensor = torch.tensor(y_main, dtype=torch.float32)
        labels_dict = {
            'string': torch.tensor(string_label, dtype=torch.long),
            'fret': torch.tensor(fret_label, dtype=torch.long),
            'pitch': torch.tensor(pitch_label, dtype=torch.long),
            'onset': torch.tensor([onset_label], dtype=torch.float32),
            'offset': torch.tensor([offset_label], dtype=torch.float32)
        }

        return audio_tensor, labels_dict
    
    def _get_dummy_labels(self):
        return{
            'string': torch.tensor(0, dtype=torch.long),
            'fret': torch.tensor(0, dtype=torch.long),
            'pitch': torch.tensor(0, dtype=torch.long),
            'onset': torch.tensor([0.0], dtype=torch.float32),
            'offset': torch.tensor([0.0], dtype=torch.float32)
        }