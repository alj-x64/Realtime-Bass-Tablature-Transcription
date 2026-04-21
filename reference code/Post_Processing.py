import torch
import numpy as np

class RealTimeHMM:
    def __init__(self, num_classes, stay_prob=0.85):
        """
        Hidden Markov Model (HMM) para sa Temporal Smoothing.
        """
        self.num_classes = num_classes
        self.transition_matrix = np.full((num_classes, num_classes), (1.0 - stay_prob) / (max(1, num_classes - 1)))
        np.fill_diagonal(self.transition_matrix, stay_prob)
        self.prev_probs = np.ones(num_classes) / num_classes

    def step(self, cnn_emission_probs):
        if torch.is_tensor(cnn_emission_probs):
            cnn_emission_probs = cnn_emission_probs.cpu().detach().numpy().flatten()
            
        prior = np.dot(self.prev_probs, self.transition_matrix)
        smoothed_probs = prior * cnn_emission_probs
        
        prob_sum = np.sum(smoothed_probs)
        if prob_sum > 0:
            smoothed_probs /= prob_sum
        else:
            smoothed_probs = np.ones(self.num_classes) / self.num_classes
            
        self.prev_probs = smoothed_probs
        return np.argmax(smoothed_probs), smoothed_probs

class BassTranscriptionDecoder:
    def __init__(self, onset_threshold=0.6, offset_threshold=0.6):
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        
        self.is_note_active = False
        self.current_note_data = {}
        
        # HMM Initialization
        self.hmm_string = RealTimeHMM(num_classes=5, stay_prob=0.90)  
        self.hmm_fret = RealTimeHMM(num_classes=14, stay_prob=0.90)   
        self.hmm_pitch = RealTimeHMM(num_classes=64, stay_prob=0.90)  
        
        # ========================================================
        # LABEL MAPPERS (Ito yung sumasagot sa tanong mo, beh!)
        # Dito natin tina-translate ang AI Index -> Human Readable
        # ========================================================
        
        # STRING MAP: Index 0 ay Silence. Index 1-4 ay physical strings.
        self.string_map = {
            0: "Silence", 1: "1st String (G)", 2: "2nd String (D)", 
            3: "3rd String (A)", 4: "4th String (E)"
        }
        
        # FRET MAP: Base sa diagram mo (S, 0, 1, ... 12)
        self.fret_map = {0: "Silence"}
        for i in range(1, 14):
            self.fret_map[i] = f"Fret {i-1}" if i > 1 else "Open String (0)"
            
        # PITCH MAP: (Estimate for 64 classes). Index 0 = Silence. 
        # MIDI 28 = E1 (Lowest Bass Note). Aabot hanggang MIDI 90.
        self.pitch_map = {0: "Silence"}
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for i in range(1, 64):
            midi_val = 27 + i # Assuming Index 1 = MIDI 28 (E1)
            octave = (midi_val // 12) - 1
            note = note_names[midi_val % 12]
            self.pitch_map[i] = f"{note}{octave} (MIDI {midi_val})"

    def decode_cnn_output(self, predictions):
        """
        Ide-decode ang AI tensor papuntang String, Fret, at Pitch text.
        """
        out_string, out_fret, out_pitch, out_onset, out_offset = predictions
        
        # 1. HMM TEMPORAL SMOOTHING (Returns indices)
        p_string_idx, _ = self.hmm_string.step(out_string)
        p_fret_idx, _ = self.hmm_fret.step(out_fret)
        p_pitch_idx, _ = self.hmm_pitch.step(out_pitch)
        
        # 2. TRANSLATE INDEX TO ACTUAL VALUES
        actual_string = self.string_map.get(int(p_string_idx), "Unknown")
        actual_fret = self.fret_map.get(int(p_fret_idx), "Unknown")
        actual_pitch = self.pitch_map.get(int(p_pitch_idx), "Unknown")
        
        # 3. ONSET / OFFSET THRESHOLDING
        onset_detected = out_onset.item() > self.onset_threshold
        offset_detected = out_offset.item() > self.offset_threshold
        
        note_event = None

        # 4. STATE MACHINE LOGIC
        if onset_detected and not self.is_note_active:
            # Kung ang HMM ay nag-predict ng "Silence" kahit may onset trigger, 
            # pwedeng ghost note ito. I-lo-log pa rin natin.
            self.is_note_active = True
            
            self.current_note_data = {
                "event": "ONSET",
                "string_raw_idx": int(p_string_idx), # Keep raw index for math/UI
                "fret_raw_idx": int(p_fret_idx),
                "decoded_string": actual_string,     # Human readable (ex: "4th String (E)")
                "decoded_fret": actual_fret,         # Human readable (ex: "Fret 5")
                "decoded_pitch": actual_pitch
            }
            note_event = self.current_note_data
            
        elif offset_detected and self.is_note_active:
            self.is_note_active = False
            note_event = {
                "event": "OFFSET",
                "decoded_string": self.current_note_data.get("decoded_string"),
                "decoded_fret": self.current_note_data.get("decoded_fret")
            }
            self.current_note_data = {}
            
        return note_event