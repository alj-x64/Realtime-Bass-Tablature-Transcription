import os
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
import scipy.signal

class DatasetAugmentation:
    def __init__(self, sr = 22050):
        self.sr = sr

    def _get_audio_path(self, root_dir, filename):
        for root, _, files in os.walk(root_dir):
            if filename in files:
                return os.path.join(root, filename)
        return None
    
    def inject_noise(self, y):
        t = np.arange(len(y)) / self.sr
        hum_freq = np.random.choice([50, 60])
        hum_strength = np.random.uniform(0.1, 0.3)

        hum = (
            hum_strength * np.sin(2 * np.pi * hum_freq * t) +
            0.5 * hum_strength * np.sin(2 * np.pi * 2 * hum_freq * t) +
            0.25 * hum_strength * np.sin(2 * np.pi * 3 * hum_freq * t)
        )

        noise = np.random.normal(0, 0.01, len(y))
        noise = scipy.signal.lfilter([1], [1, -0.95], noise)
        noise = noise / np.max(np.abs(noise))
        hiss = 0.01 * noise

        mod = 1 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
        hiss = hiss * 0.3 * mod

        spikes = np.random.choice([0, 1], size=len(y), p=[0.998, 0.002])
        spike_noise = spikes * np.random.uniform(-0.3, 0.3, len(y))
        hiss = hiss + spike_noise

        env = np.abs(y)
        env = env / np.max(env)
        hiss = 0.005 * hiss * (1 - env + 0.2)

        y = y / np.max(np.abs(y))
        y_noisy = y + hum + 0.2 * hiss

        max_val = np.max(np.abs(y_noisy))
        if max_val > 0:
            y_noisy = y_noisy / max_val

        return y_noisy
    
    def apply_eq(self, y):
        eq_mode = np.random.choice(['bass_boost', 'treble_boost', 'muffled'])

        if eq_mode == 'bass_boost':
            b, a = scipy.signal.butter(2, 150 / (self.sr /2), btype='lowpass')
            y_filtered = scipy.signal.lfilter(b, a, y)
            y_eq = 0.5 * y + 2 * y_filtered
        elif eq_mode == 'treble_boost':
            b, a = scipy.signal.butter(2, 5000 / (self.sr /2), btype='highpass')
            y_filtered = scipy.signal.lfilter(b, a, y)
            y_eq = 0.5 * y + 2 * y_filtered
        else:
            b, a = scipy.signal.butter(2, 500 / (self.sr / 2), btype='lowpass')
            y_eq = scipy.signal.lfilter(b, a, y)
        
        max_val = np.max(np.abs(y_eq))
        if max_val > 0:
            y_eq = y_eq / max_val
        return y_eq

    def concantenate_notes(self, y_main, annotations, root_dir, num_append):
        appended_labels = []
        y_concat, _ = librosa.effects.trim(np.copy(y_main), top_db=30)

        for _ in range(num_append): 
            rand_index = np.random.randint(0, len(annotations))
            bg_row = annotations.iloc[rand_index]
            bg_path = self._get_audio_path(root_dir, bg_row['filename'])

            if bg_path is None:
                print(f"File not found: {bg_path}")
                continue

            try:
                y_bg, _ = librosa.load(bg_path, sr=self.sr)

                onset = float(bg_row.get('onset', bg_row.get('onset', 0.0)))
                offset = float(bg_row.get('offset', bg_row.get('offset', 0.0)))
                pre_onset_buffer = 0.05

                start_sample = max(0, int((onset - pre_onset_buffer) * self.sr))
                y_bg_tail = y_bg[start_sample:]
                y_bg_tail, _ = librosa.effects.trim(y_bg_tail, top_db=30)

                if len(y_bg_tail) < 1000:
                    continue

                y_concat = np.atleast_1d(y_concat)
                y_bg_tail = np.atleast_1d(y_bg_tail)

                current_length = len(y_concat) / self.sr 

                actual_pre_onset = start_sample / self.sr
                time_to_onset = onset - actual_pre_onset

                new_onset = current_length + time_to_onset
                new_offset = current_length + (offset - actual_pre_onset)

                appended_labels.append({
                    'pitch': bg_row.get('pitch', 'Unknown'),
                    'string': bg_row.get('string', 0),
                    'fret': bg_row.get('fret', 0),
                    'onset': round(new_onset, 3),
                    'offset': round(new_offset, 3)
                })

                crossfade = min(100, len(y_concat), len(y_bg_tail))
                if crossfade > 10:
                    fade_out = np.linspace(1, 0, crossfade)
                    fade_in = np.linspace(0, 1, crossfade)
                    
                    y_concat_end = y_concat[-crossfade:]
                    y_bg_start = y_bg_tail[:crossfade]

                    y_concat[-crossfade:] = (y_concat_end * fade_out) + (y_bg_start * fade_in)
                    y_concat = np.concatenate((y_concat, y_bg_tail[crossfade:]))
                else:
                    y_concat = np.concatenate((y_concat, y_bg_tail))

            except Exception as e:
                print(f"Warning: {e}")

            if isinstance(y_concat, np.ndarray) and y_concat.size > 0:
                max_val = np.max(np.abs(y_concat))
                if max_val > 0:
                    y_concat = y_concat / max_val
            
        return y_concat, appended_labels

def generate_dataset(orig_csv, orig_root, dest_root, dest_csv):
    if not os.path.exists(dest_root):
        os.makedirs(dest_root)
    
    annotation = pd.read_csv(orig_csv)
    augmentor = DatasetAugmentation(sr = 22050)

    new_annotations = []
    total_files = len(annotation)

    print(f"Starting dataset augmentation ({total_files} files to augment)")

    for idx, row, in annotation.iterrows():
        orig_filename = row['filename']

        orig_path = None
        for root, _, files in os.walk(orig_root):
            if orig_filename in files:
                orig_path = os.path.join(root, orig_filename)
                break
        
        if not orig_path:
            continue

        try:
            y, sr = librosa.load(orig_path, sr= 22050)

            sf.write(os.path.join(dest_root, orig_filename), y, sr)
            new_annotations.append(row.to_dict())

            y_noise = augmentor.inject_noise(np.copy(y))
            noise_filename = orig_filename.replace('.wav', '_NOISE.wav')
            sf.write(os.path.join(dest_root, noise_filename), y_noise, sr)

            noise_row = row.to_dict()
            noise_row['filename'] = noise_filename
            if 'filepath' in noise_row:
                noise_row['filepath'] = os.path.join(dest_root, noise_filename).replace("\\", "/")
            new_annotations.append(noise_row)

            y_eq = augmentor.apply_eq(np.copy(y))
            eq_filename = orig_filename.replace('.wav', '_EQ.wav')
            sf.write(os.path.join(dest_root, eq_filename), y_eq, sr)

            eq_row = row.to_dict()
            eq_row['filename'] = eq_filename
            if 'filepath' in eq_row:
                eq_row['filepath'] = os.path.join(dest_root, eq_filename).replace("\\", "/")
            new_annotations.append(eq_row)

            """ num_to_append = np.random.choice([3,4,5])
            y_concat, extra_labels = augmentor.concantenate_notes(np.copy(y), annotation, orig_root, num_append=num_to_append)

            if len(extra_labels) > 0:
                concat_filename = orig_filename.replace('.wav', '_CONCAT.wav')
                sf.write(os.path.join(dest_root, concat_filename), y_concat, sr)

                concat_row_main = row.to_dict()
                concat_row_main['filename'] = concat_filename
                if 'filepath' in concat_row_main:
                    concat_row_main['filepath'] = os.path.join(dest_root, concat_filename).replace("\\", "/")
                new_annotations.append(concat_row_main)

                for extra in extra_labels:
                    extra_row = row.to_dict()

                    if 'filepath' in extra_row:
                        extra_row['filepath'] = os.path.join(dest_root, concat_filename).replace("\\", "/")
                    extra_row['filename'] = concat_filename
                    extra_row['pitch'] = extra['pitch']
                    extra_row['string'] = extra['string']
                    extra_row['fret'] = extra['fret']

                    extra_row['onset'] = extra['onset']
                    extra_row['offset'] = extra['offset']


                    new_annotations.append(extra_row)
 """
        except Exception as e:
            print(f"Failed to augment {orig_filename}: {e}")

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1} / {total_files} files...")

    new_annotation = pd.DataFrame(new_annotations)
    new_annotation.to_csv(dest_csv, index=False)

    print(f"Augmentation Complete \nOriginal Files: {total_files} \nTotal rows added: {len(new_annotation)} \nMaster dataset saved to: {dest_csv}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("[SYSTEM BOOT] Pinapatakbo ang datasetAugmentation.py...")
    print(f"[DIRECTORY CHECK] Ang Current Working Directory mo ay: {os.getcwd()}")
    print("="*50 + "\n")

    ORIGINAL_CSV = "databases/dataset_labels.csv"
    ORIGINAL_DIR = "IDMT-SMT-BASS"

    AUGMENTED_DIR = "IDMT-SMT-BASS/Augmented"
    AUGMENTED_CSV = "databases/augmented_dataset_labels.csv"

    generate_dataset(ORIGINAL_CSV, ORIGINAL_DIR, AUGMENTED_DIR, AUGMENTED_CSV)