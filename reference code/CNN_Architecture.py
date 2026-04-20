import torch
import torch.nn as nn

class DynamicTabCNN(nn.Module):
    def __init__(self, config, input_shape=(1, 630, 18)):
        """
        config: Dictionary na galing sa HPO na naglalaman ng hyperparameters.
        input_shape: (channels, bins, frames)
        """
        super(DynamicTabCNN, self).__init__()
        
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
        # Note: Pinanatili ko ang progressive Max Pooling dito kasi kapag 
        # isang bagsak lang sa dulo ang pooling (tulad sa drawing), sasabog 
        # ang RAM ng Jetson Nano mo sa laki ng flattened dimension.
        layers = []
        in_channels = input_shape[0] 
        
        print(f"[MODEL BUILDER] Creating CNN: {conv_layers} Layers | {filter_layers} Filters | {kernel_size}x{kernel_size} Kernel")
        
        for i in range(conv_layers):
            layers.append(nn.Conv2d(in_channels=in_channels, 
                                    out_channels=filter_layers, 
                                    kernel_size=kernel_size, 
                                    padding='same'))
            layers.append(activation_fn)
            layers.append(nn.Dropout2d(p=self.dropout_rate))
            layers.append(nn.MaxPool2d(kernel_size=(2, 2))) 
            
            in_channels = filter_layers 
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # 3. Dynamic Flattening Calculator
        dummy_input = torch.zeros(1, *input_shape) 
        with torch.no_grad():
            dummy_output = self.feature_extractor(dummy_input)
            
        self.flattened_size = dummy_output.numel() 
        print(f"[MODEL BUILDER] Flattened Dimension: {self.flattened_size}")
        
        # 4. SHARED DENSE LAYER (128 x 1 base sa diagram)
        self.shared_dense = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            activation_fn,
            nn.Dropout(self.dropout_rate)
        )
        
        # ========================================================
        # THE 5 HEADS (Strictly based on CNN Architecture.jpg)
        # Bawat head ay may sari-sariling Dense 1, Dense 2, at Dense 3
        # ========================================================

        # --- A. ONSET HEAD ---
        # Diagram: Dense 1 (128) -> Dense 2 (630) -> Output (1)
        self.head_onset = nn.Sequential(
            nn.Linear(128, 128),
            activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 630),
            activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(630, 1)
        )

        # --- B. PITCH (Note Played) HEAD ---
        # Diagram: Dense 1 (128) -> Dense 2 (630) -> Dense 3 (64) -> Output (64)
        self.head_pitch = nn.Sequential(
            nn.Linear(128, 128),
            activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 630),
            activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(630, 64) # Final output is 64 classes
        )

        # --- C. FRET NUMBER HEAD ---
        # Diagram: Dense 1 (128) -> Dense 2 (64) -> Dense 3 (14) -> Output (14)
        # Mga klase: S, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
        self.head_fret = nn.Sequential(
            nn.Linear(128, 128),
            activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 14) # Final output is 14 classes
        )

        # --- D. STRING NUMBER HEAD ---
        # Diagram: Dense 1 (128) -> Dense 2 (64) -> Dense 3 (5) -> Output (5)
        # Mga klase: 0, 1, 2, 3, 4
        self.head_string = nn.Sequential(
            nn.Linear(128, 128),
            activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 5) # Final output is 5 classes
        )

        # --- E. OFFSET HEAD ---
        # Diagram: Dense 1 (128) -> Dense 2 (630) -> Output (1)
        self.head_offset = nn.Sequential(
            nn.Linear(128, 128),
            activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 630),
            activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(630, 1)
        )

    def forward(self, x):
        """
        PyTorch quirk info beh: Hindi natin inilagay ang Softmax() dito sa forward pass 
        para sa Fret, Pitch, at String. Bakit? Dahil gagamit tayo ng nn.CrossEntropyLoss() 
        sa training, at awtomatiko na nitong inia-apply ang LogSoftmax sa loob. 
        Kung dodoblehin natin ang Softmax, masisira ang math mo at hindi matututo ang AI.
        (Pero ang Sigmoid sa Onset at Offset ay nandito kasi binary sila).
        """
        # Feature Extraction
        x = self.feature_extractor(x)
        
        # Flattening (batch_size, flattened_size)
        x = x.view(x.size(0), -1) 
        
        # Shared Dense Layer
        x = self.shared_dense(x)
        
        # Multi-Task Heads Execution
        out_string = self.head_string(x) # Raw Logits
        out_fret = self.head_fret(x)     # Raw Logits
        out_pitch = self.head_pitch(x)   # Raw Logits
        
        # Sigmoid Applied direct as per diagram for Binary outputs (0.00 - 1.00)
        out_onset = torch.sigmoid(self.head_onset(x))
        out_offset = torch.sigmoid(self.head_offset(x))
        
        return out_string, out_fret, out_pitch, out_onset, out_offset