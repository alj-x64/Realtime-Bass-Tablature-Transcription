import os
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class ISBDataset(Dataset):
    def __init__(self, csv_file, root_dir, audio_length=4608, sr=22050):
        """
        Pure Data I/O Loader para sa GPU Training.
        Tatanggapin na lang nito ang files galing sa Augmented Folder mo.
        Wala nang heavy DSP processing dito para bumilis ang PyTorch loop.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.audio_length = audio_length
        self.sr = sr
        
        # PITCH MAPPING
        self.pitch_to_idx = {"Silence": 0}
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for i in range(1, 64):
            midi_val = 27 + i
            octave = (midi_val // 12) - 1
            note = notes[midi_val % 12]
            pitch_str = f"{note}{octave}"
            self.pitch_to_idx[pitch_str] = i

    def __len__(self):
        return len(self.annotations)
        
    def _get_audio_path(self, file_name):
        for root, _, files in os.walk(self.root_dir):
            if file_name in files:
                return os.path.join(root, file_name)
        return None

    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        audio_path = self._get_audio_path(row['filename'])
                
        if audio_path is None:
            return torch.zeros(self.audio_length), self._get_dummy_labels()

        try:
            onset_time = row['onset_time']
            offset_time = row['offset_time']
            duration = librosa.get_duration(path=audio_path)
            
            # Dynamic State Sampling (Targeting the 209ms context frame)
            scenario = np.random.choice(['onset', 'offset', 'sustain', 'silence'])
            
            if scenario == 'onset':
                target_time = onset_time
                onset_lbl, offset_lbl, is_active = 1.0, 0.0, True
            elif scenario == 'offset':
                target_time = offset_time
                onset_lbl, offset_lbl, is_active = 0.0, 1.0, False
            elif scenario == 'sustain' and (offset_time - onset_time > 0.3):
                target_time = np.random.uniform(onset_time + 0.1, offset_time - 0.1)
                onset_lbl, offset_lbl, is_active = 0.0, 0.0, True
            else: 
                # Silence sampling
                if onset_time > 0.4 and np.random.rand() > 0.5:
                    target_time = np.random.uniform(0, onset_time - 0.2)
                elif duration > offset_time + 0.4:
                    target_time = np.random.uniform(offset_time + 0.2, duration)
                else:
                    target_time = duration 
                onset_lbl, offset_lbl, is_active = 0.0, 0.0, False

            # Extract Pure 4608-sample Frame directly from disk (I/O Optimization)
            # Ginagamit ang offset/duration parameter ng librosa para hindi na i-load sa RAM ang buong file
            start_sample = max(0, int(target_time * self.sr) - (self.audio_length // 2))
            y_main, _ = librosa.load(audio_path, sr=self.sr, offset=start_sample/self.sr, duration=self.audio_length/self.sr)
            
            # Constant padding kung kulang ang na-load na size (madalas mangyari sa dulo ng audio file)
            if len(y_main) < self.audio_length:
                y_main = np.pad(y_main, (0, self.audio_length - len(y_main)), mode='constant')
            else:
                y_main = y_main[:self.audio_length]

        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            y_main = np.zeros(self.audio_length, dtype=np.float32)
            is_active, onset_lbl, offset_lbl = False, 0.0, 0.0

        # Label Parsing
        if is_active:
            string_label = int(row['string']) 
            fret_label = int(row['fret']) + 1 
            pitch_label = self.pitch_to_idx.get(row['pitch'], 0)
        else:
            string_label = 0 
            fret_label = 0   
            pitch_label = 0  

        # Tensor Conversion
        audio_tensor = torch.tensor(y_main, dtype=torch.float32)
        labels_dict = {
            'string': torch.tensor(string_label, dtype=torch.long),
            'fret': torch.tensor(fret_label, dtype=torch.long),
            'pitch': torch.tensor(pitch_label, dtype=torch.long),
            'onset': torch.tensor([onset_lbl], dtype=torch.float32),   
            'offset': torch.tensor([offset_lbl], dtype=torch.float32)  
        }

        return audio_tensor, labels_dict

    def _get_dummy_labels(self):
        """Fallback kung hindi mahanap ang file sa disk."""
        return {
            'string': torch.tensor(0, dtype=torch.long),
            'fret': torch.tensor(0, dtype=torch.long),
            'pitch': torch.tensor(0, dtype=torch.long),
            'onset': torch.tensor([0.0], dtype=torch.float32),
            'offset': torch.tensor([0.0], dtype=torch.float32)
        }