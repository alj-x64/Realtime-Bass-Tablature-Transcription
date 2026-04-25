import pandas as pd
import os
import librosa
import re
import numpy as np
import csv

def midi_to_pitch(midi_note):
    notes = ['C','C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note//12) - 1
    note = notes[midi_note % 12]
    return f"{note}{octave}"

def generate_annotations(dataset_directory, output_csv="dataset_labels.csv"):
     # MIDI values for open string
     string_base_midi = {1: 16,
                         2: 21,
                         3: 26,
                         4: 31}
     
     filepaths = []
     for root, dirs, files in os.walk(dataset_directory):
          for file in files:
               if file.lower().endswith('.wav'):
                    filepaths.append(os.path.join(root, file))

     total_files = len(filepaths)
     print(f"Found {total_files} audio files. Process starts.")

     if total_files == 0:
          print("No .wav files found")
          return
     
     with open(output_csv, mode='w', newline='') as csv_file:
          fieldnames = ['filepath',
                        'filename', 
                        'pitch', 
                        'string', 
                        'fret', 
                        'onset', 
                        'offset']
          
          writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
          writer.writeheader()

          for idx, filepath in enumerate(filepaths):
               filename = os.path.basename(filepath)

               match_four = re.search(r'BS_\d+_EQ_\d+_[A-Za-z]+_[A-Za-z]+_[A-Za-z]+_[A-Za-z]+_(\d+)_(\d+)\.wav', 
                                 filename, 
                                 re.IGNORECASE)

               match_normal = re.search(r'BS_\d+_EQ_\d+_[A-Za-z]+_[A-Za-z]++_(\d+)_(\d+)\.wav', 
                                 filename, 
                                 re.IGNORECASE)
               
               match = match_normal or match_four

               if not match:
                    print(f"[SKIP] Cannot parse IDMT format of {filename}")
                    continue

               string_num = int(match.group(1))
               fret_num = int(match.group(2))

               if string_num not in string_base_midi:
                    print(f"[SKIP] Invalid string number ({string_num}) in file: {filename}")
                    continue

               actual_midi = string_base_midi[string_num] + fret_num
               note_pitch = midi_to_pitch(actual_midi)

               try:
                    y, sr = librosa.load(filepath, sr= 22500)

                    # ONSET DETECTION
                    onset_frames = librosa.onset.onset_detect(y = y,
                                                              sr = sr,
                                                              backtrack = True)
                    if len(onset_frames) > 0:
                         onset_time = librosa.frames_to_time(onset_frames[0], 
                                                             sr=sr)
                    else:
                         onset_time = 0.0

                    # OFFSET DETECTION
                    intervals = librosa.effects.split(y, top_db=30)

                    if len(intervals) > 0:
                         offset_sample = intervals[-1][1]
                         offset_time = librosa.samples_to_time(offset_sample, 
                                                               sr=sr)
                    else:
                         offset_time = librosa.get_duration(y=y, 
                                                            sr=sr)
                         
                    if offset_time <= onset_time:
                         offset_time = librosa.get_duration(y=y,
                                                            sr=sr)
                         
               except Exception as e:
                    print(f"[ERROR] Problem processing {filename}: {e}")
                    continue

               writer.writerow({
                    'filepath': filepath.replace("\\", "/"),
                    'filename': filename,
                    'pitch': note_pitch,
                    'string': string_num,
                    'fret': fret_num,
                    'onset': round(onset_time, 3),
                    'offset': round(offset_time, 3)
               })

               if (idx + 1) % 50 == 0:
                    print(f"Processed {idx + 1}/{total_files} files... ")

     print(f"DONE! Annotations saved at {output_csv}")



if __name__ == "__main__":
     TARGET_FOLDER = "IDMT-SMT-BASS"

     if not os.path.exists(TARGET_FOLDER):
          os.makedirs(TARGET_FOLDER)
          print(f"No folder detected. {TARGET_FOLDER} created")

     else: generate_annotations(dataset_directory=TARGET_FOLDER, output_csv="databases/dataset_labels.csv")