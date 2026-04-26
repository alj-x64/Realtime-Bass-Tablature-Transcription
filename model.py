import torch
import torch.nn as nn
import nnAudio.features
import torch.nn.functional as F

SR = 22050
HOP_LENGTH = 256

CQT_BINS = 189
INPUT_FRAMES = 18

TAB_CLASSES = (1 + 12) * 4 #(1 OPEN + 12 FRETS) * 4 STRINGS
PITCH_BINS = CQT_BINS
TECHNIQUE_CLASS = 1 #finger-style

class BassTranscriptionCNN(nn.Module):
    def __init__(self, config, audio_length = int(SR)):
        super().__init__()

        #   INPUT LAYER
        self.cqt = nnAudio.features.CQT(
            sr = SR,
            hop_length = HOP_LENGTH,
            fmin = 41.2, #E1
            n_bins = CQT_BINS,
            bins_per_octave = 36,
            trainable = False,
            output_format = 'Magnitude',
            pad_mode='reflect'
        )

        self.audio_length = audio_length
        self.expected_min = self.audio_length
        #   HPO PARAMETER EXTRACTION
        #   Values specified are just fallback values to prevent crashes or errors during training

        conv_layers = config.get('convolution_layers', 3)
        filter_layers = config.get('filter_layers', 32)
        kernel_size = config.get('filter_size', 3)
        dropout_rate = config.get('dropout_rate', 0.25)

        activation_config = config.get('activation_function', 'ReLU')
        if activation_config == 'ReLU':
            activation_function = nn.ReLU()
        elif activation_config == 'Tanh':
            activation_function = nn.Tanh()
        else:
            activation_function = nn.ELU()

        # FEATURE EXTRACTION
        layers = []
        in_channel = 1

        pad_size = kernel_size // 2

        for i in range(conv_layers):
            layers.append(nn.Conv2d(in_channels = in_channel,
                                     out_channels = filter_layers,
                                     kernel_size = kernel_size,
                                     padding = pad_size,
                                     padding_mode = 'reflect'))
            layers.append(activation_function)
            layers.append(nn.Dropout2d(p=dropout_rate))

            in_channel = filter_layers
        
        layers.append(nn.MaxPool2d(kernel_size=(2,2)))

        self.feature_extractor = nn.Sequential(*layers)

        #   FLATTENING PRIOR TO SHARED DENSE LAYER
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        #self.flattened = filter_layers

        #   SHARED DENSE LAYER
        self.shared_dense = nn.Sequential(nn.Linear(filter_layers, 128),
                                          nn.ReLU(),
                                          nn.Dropout(dropout_rate))
        
        #   MULTI-TASK LEARNING HEADS
        #   HEAD 1 - ONSET
        self.head_onset = nn.Sequential(nn.Linear(128, 128),
                                        nn.ReLU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(128, CQT_BINS),
                                        nn.ReLU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(CQT_BINS, 1))
        #   HEAD 2 - PITCH
        self.head_pitch = nn.Sequential(nn.Linear(128, 128),
                                        nn.ReLU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(128, 64))
        #   HEAD 3 - FRET
        self.head_fret = nn.Sequential(nn.Linear(128, 128),
                                       nn.ReLU(),
                                       nn.Dropout(dropout_rate),
                                       nn.Linear(128, 64),
                                       nn.ReLU(),
                                       nn.Dropout(dropout_rate),
                                       nn.Linear(64, 14))
        #   HEAD 4 - STRING
        self.head_string = nn.Sequential(nn.Linear(128, 128),
                                         nn.ReLU(),
                                         nn.Dropout(dropout_rate),
                                         nn.Linear(128, 64),
                                         nn.ReLU(),
                                         nn.Dropout(dropout_rate),
                                         nn.Linear(64, 5))
        #   HEAD 5 - OFFSET
        self.head_offset = nn.Sequential(nn.Linear(128, 128),
                                         nn.ReLU(),
                                         nn.Dropout(dropout_rate),
                                         nn.Linear(128, CQT_BINS),
                                         nn.ReLU(),
                                         nn.Dropout(dropout_rate),
                                         nn.Linear(CQT_BINS, 1))
    def _infer_flattened_shape(self):
        dummy = torch.zeros(1, self.audio_length)

        with torch.no_grad():
            x = self.cqt(dummy)
            x = self.feature_extractor(x)
            x = x.view(1, -1)

        return x.size(1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        target_len = self.audio_length

        if x.shape[-1] < target_len:
            x = F.pad(x, (0, target_len - x.shape[-1]))
        else:
            x = x[:, :target_len]
        
        x = self.cqt(x)

        if x.shape[-1] < INPUT_FRAMES:
            x = F.pad(x, (0, INPUT_FRAMES - x.shape[-1]))
        else:
            x = x[:, :, :INPUT_FRAMES]

        x = x.unsqueeze(1)

        x = self.feature_extractor(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)
        #x = x.view(x.size(0), -1)

        x = self.shared_dense(x)

        out_string = self.head_string(x)
        out_fret = self.head_fret(x)
        out_pitch = self.head_pitch(x)

        out_onset = torch.sigmoid(self.head_onset(x))
        out_offset = torch.sigmoid(self.head_offset(x))

        return out_string, out_fret, out_pitch, out_onset, out_offset