import os
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
import scipy.signal

# =======================================================================
# 1. THE PHYSICS ENGINE 
# =======================================================================
class BassAudioAugmentor:
    def __init__(self, sr=22050):
        """
        Kumpletong Audio Processing Engine para sa Data Augmentation.
        1. Noise Injection
        2. EQ Change Simulation
        3. Continuous Bassline (Direct Append - Walang Putol)
        """
        self.sr = sr

    def _get_audio_path(self, root_dir, file_name):
        for root, _, files in os.walk(root_dir):
            if file_name in files:
                return os.path.join(root, file_name)
        return None

    def inject_noise(self, y):
        """1. INDIVIDUAL RECORDING - NOISE INJECTION"""
        noise_amp = np.random.uniform(0.001, 0.015)
        noise = np.random.normal(0, noise_amp, len(y))
        y_noisy = y + noise
        
        max_val = np.max(np.abs(y_noisy))
        if max_val > 0:
            y_noisy = y_noisy / max_val
        return y_noisy

    def apply_eq_change(self, y):
        """2. INDIVIDUAL RECORDING - EQ CHANGE"""
        eq_mode = np.random.choice(['bass_boost', 'treble_boost', 'muffled'])
        
        if eq_mode == 'bass_boost':
            b, a = scipy.signal.butter(2, 250 / (self.sr / 2), btype='low')
            y_filtered = scipy.signal.lfilter(b, a, y)
            y_eq = y + 0.6 * y_filtered 
            
        elif eq_mode == 'treble_boost':
            b, a = scipy.signal.butter(2, 2000 / (self.sr / 2), btype='high')
            y_filtered = scipy.signal.lfilter(b, a, y)
            y_eq = y + 0.5 * y_filtered
            
        else: # 'muffled'
            b, a = scipy.signal.butter(2, 800 / (self.sr / 2), btype='low')
            y_eq = scipy.signal.lfilter(b, a, y)

        max_val = np.max(np.abs(y_eq))
        if max_val > 0:
            y_eq = y_eq / max_val
        return y_eq

    def concatenate_notes(self, y_main, annotations, root_dir):
        """3. MAG-CONCATENATE NG NOTES (DIRECT APPEND WALANG PUTOL)"""
        rand_idx = np.random.randint(0, len(annotations))
        bg_row = annotations.iloc[rand_idx]
        bg_path = self._get_audio_path(root_dir, bg_row['filename'])
        
        if bg_path is None:
            return y_main
            
        try:
            # 1. I-load ang buong background audio (Walang limit!)
            y_bg, _ = librosa.load(bg_path, sr=self.sr)
            
            # 2. Tanggalin ang dead air sa simula ng background note
            # Para pag dinugtong natin, papasok agad yung pangalawang nota
            yt_bg, index = librosa.effects.trim(y_bg, top_db=30)
            y_bg_trimmed = y_bg[index[0]:]
            
            # 3. DIRECT APPEND WITH CROSSFADE (Anti-Pop/Click Artifact)
            crossfade_len = 100 # ~4.5 milliseconds crossfade
            
            if len(y_main) > crossfade_len and len(y_bg_trimmed) > crossfade_len:
                fade_out = np.linspace(1, 0, crossfade_len)
                fade_in = np.linspace(0, 1, crossfade_len)
                
                # Overlap zone
                y_main[-crossfade_len:] = (y_main[-crossfade_len:] * fade_out) + (y_bg_trimmed[:crossfade_len] * fade_in)
                
                # Append ang natitirang part
                y_concat = np.concatenate((y_main, y_bg_trimmed[crossfade_len:]))
            else:
                # Kung masyadong maikli ang audio, pure append na lang
                y_concat = np.concatenate((y_main, y_bg_trimmed))

            # 4. Normalize volume
            max_val = np.max(np.abs(y_concat))
            if max_val > 0:
                y_concat = y_concat / max_val
                
            return y_concat
            
        except Exception as e:
            print(f"Splicing warning: {e}")
            return y_main

# =======================================================================
# 2. THE OFFLINE GENERATOR LOGIC
# =======================================================================
def generate_offline_dataset(orig_csv, orig_root, dest_root, dest_csv):
    if not os.path.exists(dest_root):
        os.makedirs(dest_root)
        
    df = pd.read_csv(orig_csv)
    augmentor = BassAudioAugmentor(sr=22050)
    
    new_annotations = []
    total_files = len(df)
    
    print(f"--- STARTING OFFLINE AUGMENTATION ({total_files} Original Files) ---")
    print(f"Target Destination: {dest_root}")
    
    for idx, row in df.iterrows():
        orig_filename = row['filename']
        
        orig_path = None
        for root, _, files in os.walk(orig_root):
            if orig_filename in files:
                orig_path = os.path.join(root, orig_filename)
                break
                
        if not orig_path:
            continue
            
        try:
            # Load Original Audio (Walang limit sa duration)
            y, sr = librosa.load(orig_path, sr=22050)
            
            # --- 1. SAVE ORIGINAL ---
            sf.write(os.path.join(dest_root, orig_filename), y, sr)
            new_annotations.append(row.to_dict())
            
            # --- 2. GENERATE & SAVE NOISE VERSION ---
            y_noise = augmentor.inject_noise(np.copy(y))
            noise_filename = orig_filename.replace('.wav', '_NOISE.wav')
            sf.write(os.path.join(dest_root, noise_filename), y_noise, sr)
            
            noise_row = row.to_dict()
            noise_row['filename'] = noise_filename
            new_annotations.append(noise_row)
            
            # --- 3. GENERATE & SAVE EQ VERSION ---
            y_eq = augmentor.apply_eq_change(np.copy(y))
            eq_filename = orig_filename.replace('.wav', '_EQ.wav')
            sf.write(os.path.join(dest_root, eq_filename), y_eq, sr)
            
            eq_row = row.to_dict()
            eq_row['filename'] = eq_filename
            new_annotations.append(eq_row)
            
            # --- 4. GENERATE & SAVE CONCATENATED VERSION ---
            # WALA NG PADDING. Diretsong ipinapasa ang y nang buo.
            y_concat = augmentor.concatenate_notes(np.copy(y), df, orig_root)
            
            concat_filename = orig_filename.replace('.wav', '_CONCAT.wav')
            sf.write(os.path.join(dest_root, concat_filename), y_concat, sr)
            
            concat_row = row.to_dict()
            concat_row['filename'] = concat_filename
            new_annotations.append(concat_row)
            
        except Exception as e:
            print(f"[ERROR] Failed to augment {orig_filename}: {e}")
            
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{total_files} files... Generated {(idx+1)*4} new files.")

    new_df = pd.DataFrame(new_annotations)
    new_df.to_csv(dest_csv, index=False)
    
    print(f"\n✅ OFFLINE AUGMENTATION COMPLETE!")
    print(f"Original Files: {total_files}")
    print(f"Total Files in New Dataset: {len(new_df)}")
    print(f"Saved Master CSV to: {dest_csv}")

if __name__ == "__main__":
    ORIGINAL_CSV = "bass_annotations.csv"
    ORIGINAL_DIR = "./IDMT-SMT-BASS"
    
    AUGMENTED_DIR = "./IDMT-SMT-BASS_AUGMENTED"
    AUGMENTED_CSV = "augmented_bass_annotations.csv"
    
    generate_offline_dataset(ORIGINAL_CSV, ORIGINAL_DIR, AUGMENTED_DIR, AUGMENTED_CSV)