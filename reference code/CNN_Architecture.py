import torch
import torch.nn as nn
# Import natin ang nnAudio. Siguraduhing naka-install ito (pip install nnAudio)
from nnAudio.features.cqt import CQT1992v2 

class DynamicTabCNN(nn.Module):
    def __init__(self, config, audio_length=4608):
        """
        config: Dictionary na galing sa HPO na naglalaman ng hyperparameters.
        audio_length: Raw audio buffer size (256 hop_size * 18 frames = 4608 samples)
        """
        super(DynamicTabCNN, self).__init__()
        
        # ========================================================
        # 0. THE nnAudio INPUT LAYER (Layer 0)
        # ========================================================
        # Dito natin kino-convert ang raw audio (1D) papuntang Spectrogram (2D) sa loob ng GPU.
        # NOTE: Naka-set ang n_bins sa 630 base sa CNN Architecture.jpg mo. 
        # Maaari mong i-adjust ang fmin at bins_per_octave base sa exact math ng thesis mo.
        self.cqt_layer = CQT1992v2(
            sr=22050, 
            hop_length=256, 
            fmin=32.70, # C1 note (typical starting frequency for bass/guitar)
            n_bins=630, # Base sa diagram mong 630 x 18
            bins_per_octave=105, # 6 octaves * 105 = 630 (Adjust depending on resolution)
            output_format='Magnitude', 
            trainable=False # False para static ang CQT kernel, hindi magbabago habang nagte-train
        )

        # 1. HPO Parameters Extraction
        conv_layers = config.get('conv_layers', 3)
        filter_layers = config.get('filter_layers', 32)
        kernel_size = config.get('kernel_size', 3)
        self.dropout_rate = config.get('dropout_rate', 0.25)
        
        activation_str = config.get('activation', 'ReLU')
        if activation_str == 'ReLU':
            activation_fn = nn.ReLU()
        elif activation_str == 'Tanh':
            activation_fn = nn.Tanh()
        else:
            activation_fn = nn.ELU()

        # 2. Dynamic Feature Extractor (CNN Builder)
        layers = []
        in_channels = 1 # 1 channel kasi Grayscale ang lalabas sa CQT
        
        print(f"[MODEL BUILDER] Creating CNN: {conv_layers} Layers | {filter_layers} Filters | {kernel_size}x{kernel_size} Kernel | Reflect Padding")
        
        # Engineering Workaround: PyTorch does not allow padding='same' with padding_mode='reflect'.
        # Since kernel sizes mo ay odd numbers (3, 5, 7), the formula for 'same' padding is kernel_size // 2.
        pad_size = kernel_size // 2

        for i in range(conv_layers):
            layers.append(nn.Conv2d(in_channels=in_channels, 
                                    out_channels=filter_layers, 
                                    kernel_size=kernel_size, 
                                    padding=pad_size, 
                                    padding_mode='reflect'))
            layers.append(activation_fn)
            layers.append(nn.Dropout2d(p=self.dropout_rate))
            
            in_channels = filter_layers 
            
        # NAG-IISANG MAX POOLING LAYER SA DULO
        layers.append(nn.MaxPool2d(kernel_size=(2, 2))) 
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # 3. Dynamic Flattening Calculator (Gumagamit ng raw audio dummy)
        dummy_audio = torch.zeros(1, audio_length) # [Batch=1, 4608 samples]
        with torch.no_grad():
            # I-simulate ang pagdaan ng audio sa CQT at CNN para makuha ang tamang dimension
            dummy_cqt = self.cqt_layer(dummy_audio) # Output: (1, 630, 19) depending on hop padding
            # I-trim/pad para maging sakto sa 18 frames as per your architecture diagram
            if dummy_cqt.shape[2] > 18:
                dummy_cqt = dummy_cqt[:, :, :18]
            dummy_cqt = dummy_cqt.unsqueeze(1) # Add channel: (1, 1, 630, 18)
            dummy_output = self.feature_extractor(dummy_cqt)
            
        self.flattened_size = dummy_output.numel() 
        print(f"[MODEL BUILDER] Flattened Dimension after CQT & Conv: {self.flattened_size}")
        
        # 4. SHARED DENSE LAYER (128 x 1 base sa diagram)
        self.shared_dense = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            activation_fn,
            nn.Dropout(self.dropout_rate)
        )
        
        # ========================================================
        # THE 5 HEADS (Strictly based on CNN Architecture.jpg)
        # ========================================================

        # --- A. ONSET HEAD ---
        self.head_onset = nn.Sequential(
            nn.Linear(128, 128), activation_fn, nn.Dropout(self.dropout_rate),
            nn.Linear(128, 630), activation_fn, nn.Dropout(self.dropout_rate),
            nn.Linear(630, 1)
        )

        # --- B. PITCH (Note Played) HEAD ---
        self.head_pitch = nn.Sequential(
            nn.Linear(128, 128), activation_fn, nn.Dropout(self.dropout_rate),
            nn.Linear(128, 630), activation_fn, nn.Dropout(self.dropout_rate),
            nn.Linear(630, 64) 
        )

        # --- C. FRET NUMBER HEAD ---
        self.head_fret = nn.Sequential(
            nn.Linear(128, 128), activation_fn, nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64), activation_fn, nn.Dropout(self.dropout_rate),
            nn.Linear(64, 14) 
        )

        # --- D. STRING NUMBER HEAD ---
        self.head_string = nn.Sequential(
            nn.Linear(128, 128), activation_fn, nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64), activation_fn, nn.Dropout(self.dropout_rate),
            nn.Linear(64, 5) 
        )

        # --- E. OFFSET HEAD ---
        self.head_offset = nn.Sequential(
            nn.Linear(128, 128), activation_fn, nn.Dropout(self.dropout_rate),
            nn.Linear(128, 630), activation_fn, nn.Dropout(self.dropout_rate),
            nn.Linear(630, 1)
        )

    def forward(self, x):
        """
        x shape pagpasok: (batch_size, 4608) -> Raw 1D Audio
        """
        # 1. nnAudio GPU CQT Extraction
        x = self.cqt_layer(x) # Output shape: (batch_size, 630, frames)
        
        # 2. Frame alignment (Enforcing strictly 18 frames as per your outline)
        if x.shape[2] > 18:
            x = x[:, :, :18]
        elif x.shape[2] < 18:
            # Zero-padding just in case the audio chunk was slightly short
            padding = torch.zeros(x.size(0), x.size(1), 18 - x.size(2), device=x.device)
            x = torch.cat((x, padding), dim=2)
            
        # 3. Add Channel Dimension for 2D CNN
        x = x.unsqueeze(1) # Shape becomes (batch_size, 1, 630, 18)
        
        # 4. Feature Extraction & Flattening
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1) 
        
        # 5. Shared Dense Layer
        x = self.shared_dense(x)
        
        # 6. Multi-Task Heads Execution
        out_string = self.head_string(x) 
        out_fret = self.head_fret(x)     
        out_pitch = self.head_pitch(x)   
        
        out_onset = torch.sigmoid(self.head_onset(x))
        out_offset = torch.sigmoid(self.head_offset(x))
        
        return out_string, out_fret, out_pitch, out_onset, out_offset