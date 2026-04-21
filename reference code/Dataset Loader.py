import os
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class ISBDataset(Dataset):
    def __init__(self, csv_file, root_dir, audio_length=4608, sr=22050):
        """
        Ang Tulay sa pagitan ng CSV Annotations mo at ng PyTorch Training Loop.
        """
        # Basahin ang ginawang CSV ng dataset_annotator.py
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.audio_length = audio_length
        self.sr = sr
        
        # PITCH MAPPING (Katumbas ng nasa transcription_utils.py)
        # E1 (MIDI 28) = Index 1. 0 = Silence
        self.pitch_to_idx = {"Silence": 0}
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for i in range(1, 64):
            midi_val = 27 + i
            octave = (midi_val // 12) - 1
            note = notes[midi_val % 12]
            pitch_str = f"{note}{octave}"
            self.pitch_to_idx[pitch_str] = i

    def __len__(self):
        """Sinasabi kay PyTorch kung gaano karami ang dataset."""
        return len(self.annotations)

    def __getitem__(self, index):
        """
        Dito nangyayari ang magic tuwing humihingi ang DataLoader ng isang batch.
        Kinukuha nito ang audio at kino-convert ang text labels to AI tensors.
        """
        # 1. Kunin ang impormasyon sa kasalukuyang row ng CSV
        row = self.annotations.iloc[index]
        file_name = row['filename']
        
        # Hanapin ang exact file (Dahil nested sa mga folders ang IDMT dataset)
        audio_path = None
        for root, _, files in os.walk(self.root_dir):
            if file_name in files:
                audio_path = os.path.join(root, file_name)
                break
                
        if audio_path is None:
            # Fallback kung sakaling nawawala ang file
            dummy_audio = torch.zeros(self.audio_length)
            return dummy_audio, self._get_dummy_labels()

        # 2. DYNAMIC STATE SAMPLING (Sustain, Onset, Offset, Silence)
        try:
            onset_time = row['onset_time']
            offset_time = row['offset_time']
            duration = librosa.get_duration(path=audio_path)
            
            # Randomly pick a training scenario para sa frame na 'to
            scenario = np.random.choice(['onset', 'offset', 'sustain', 'silence'])
            
            if scenario == 'onset':
                target_time = onset_time
                onset_lbl = 1.0
                offset_lbl = 0.0
                is_active = True
                
            elif scenario == 'offset':
                target_time = offset_time
                onset_lbl = 0.0
                offset_lbl = 1.0
                is_active = False # Tinuturing nating tapos na ang note pagka-offset
                
            elif scenario == 'sustain' and (offset_time - onset_time > 0.3):
                # Kumuha ng random slice sa gitna ng note (kung mahaba ang note)
                target_time = np.random.uniform(onset_time + 0.1, offset_time - 0.1)
                onset_lbl = 0.0
                offset_lbl = 0.0
                is_active = True
                
            else: 
                # SILENCE: Kumuha ng slice bago magsimula o pagkatapos matapos ang note
                if onset_time > 0.4 and np.random.rand() > 0.5:
                    target_time = np.random.uniform(0, onset_time - 0.2)
                elif duration > offset_time + 0.4:
                    target_time = np.random.uniform(offset_time + 0.2, duration)
                else:
                    target_time = duration # Fallback sa pinakadulo
                onset_lbl = 0.0
                offset_lbl = 0.0
                is_active = False

            # Kunin ang saktong 4608 samples palibot sa target_time
            start_sample = max(0, int(target_time * self.sr) - (self.audio_length // 2))
            y, _ = librosa.load(audio_path, sr=self.sr, offset=start_sample/self.sr, 
                                duration=self.audio_length/self.sr)
            
            # Padding kung sakaling bitin ang audio sa dulo ng kanta
            if len(y) < self.audio_length:
                pad_width = self.audio_length - len(y)
                y = np.pad(y, (0, pad_width), mode='constant')
            else:
                y = y[:self.audio_length]
                
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            y = np.zeros(self.audio_length, dtype=np.float32)
            is_active, onset_lbl, offset_lbl = False, 0.0, 0.0

        # 3. Label Extraction & Translation (Text -> Math Index)
        # Nakadepende na ngayon ang labels kung 'active' ba yung audio frame na nahiwa
        if is_active:
            string_label = int(row['string']) 
            fret_label = int(row['fret']) + 1 
            pitch_label = self.pitch_to_idx.get(row['pitch'], 0)
        else:
            string_label = 0 # Index 0 means Silence/No String
            fret_label = 0   
            pitch_label = 0  

        # 4. I-pack lahat sa PyTorch Tensors
        audio_tensor = torch.tensor(y, dtype=torch.float32)
        
        labels_dict = {
            'string': torch.tensor(string_label, dtype=torch.long),
            'fret': torch.tensor(fret_label, dtype=torch.long),
            'pitch': torch.tensor(pitch_label, dtype=torch.long),
            'onset': torch.tensor([onset_lbl], dtype=torch.float32),   
            'offset': torch.tensor([offset_lbl], dtype=torch.float32)  
        }

        return audio_tensor, labels_dict

    def _get_dummy_labels(self):
        """Fallback function kung nawawala ang data."""
        return {
            'string': torch.tensor(0, dtype=torch.long),
            'fret': torch.tensor(0, dtype=torch.long),
            'pitch': torch.tensor(0, dtype=torch.long),
            'onset': torch.tensor([0.0], dtype=torch.float32),
            'offset': torch.tensor([0.0], dtype=torch.float32)
        }