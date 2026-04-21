import torch
import torch.nn as nn
import nnAudio.features

SR = 22050
HOP_LENGTH = 256

CQT_BINS = 630
INPUT_FRAMES = 18

TAB_CLASSES = (1 + 12) * 4 #(1 OPEN + 12 FRETS) * 4 STRINGS
PITCH_BINS = CQT_BINS
TECHNIQUE_CLASS = 1 #finger-style

class BassTranscriptionCNN(nn.Module):
    def __init__(self, config, audio_length = HOP_LENGTH * INPUT_FRAMES):
        super(BassTranscriptionCNN, self).__init__()

        #   INPUT LAYER
        self.cqt = nnAudio.features.CQT(
            sr = SR,
            hop_length = HOP_LENGTH,
            fmin = 41.2, #E1
            n_bins = CQT_BINS,
            bins_per_octave = 120,
            trainable = False,
            output_format = 'Magnitude'
        )

        #   HPO PARAMETER EXTRACTION
        #   Values specified are just fallback values to prevent crashes or errors during training

        conv_layers = config.get('convolution_layers', 3)
        filter_layers = config.get('filter_layers', 32)
        kernel_size = config.get('filter_size', 3)
        dropout_rate = config.get('dropout_rate', 0.25)

        activation_config = config.get('activation', 'ReLU')
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
            layers.apppend(nn.Conv2d(in_channels = in_channel,
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
        dummy_audio = torch.zeros(1, audio_length)
        with torch.no_grad():
            dummy_cqt = self.cqt(dummy_audio)

            if dummy_cqt.shape[2] > 18:
                dummy_cqt = dummy_cqt[:, :, :18]
            dummy_cqt = dummy_cqt.unsqueeze(1)
            dummy_out = self.feature_extractor(dummy_cqt)
        
        self.flattened = dummy_out.numel()

        #   SHARED DENSE LAYER
        self.shared_dense = nn.Sequential(nn.Linear(self.flattened, 128),
                                          nn.ReLU,
                                          nn.Dropout(dropout_rate))
        
        #   MULTI-TASK LEARNING HEADS
        #   HEAD 1 - ONSET
        self.head_onset = nn.Sequential(nn.Linear(128, 128),
                                        nn.ReLU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(128, 630),
                                        nn.ReLU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(630, 1))
        #   HEAD 2 - PITCH
        self.head_pitch = nn.Sequential(nn.Linear(128, 128),
                                        nn.ReLU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(128, 630),
                                        nn.ReLU(),
                                        nn.Dropout(dropout_rate),
                                        nn.Linear(630, 64),
                                        nn.ReLU(),
                                        nn.Dropout(dropout_rate))
        #   HEAD 3 - FRET
        self.head_fret = nn.Sequential(nn.Linear(128, 128),
                                       nn.ReLU(),
                                       nn.Dropout(dropout_rate),
                                       nn.Linear(128, 64),
                                       nn.ReLU(),
                                       nn.Dropout(dropout_rate),
                                       nn.Linear(64, 14),
                                       nn.ReLU(),
                                       nn.Dropout(dropout_rate))
        #   HEAD 4 - STRING
        self.head_string = nn.Sequential(nn.Linear(128, 128),
                                         nn.ReLU(),
                                         nn.Dropout(dropout_rate),
                                         nn.Linear(128, 64),
                                         nn.ReLU(),
                                         nn.Dropout(dropout_rate),
                                         nn.Linear(64, 5),
                                         nn.ReLU(),
                                         nn.Dropout(dropout_rate))
        #   HEAD 5 - OFFSET
        self.head_offset = nn.Sequential(nn.Linear(128,128),
                                         nn.ReLU(),
                                         nn.Dropout(dropout_rate),
                                         nn.Linear(128, 630),
                                         nn.ReLU(),
                                         nn.Dropout(dropout_rate))
    
    def forward(self, x):
        x = self.cqt(x)

        if x.shape[2] > 18:
            x = x[:, :, :18]
        elif x.shape[2] < 18:
            padding = torch.zeros(x.size(0), x.size(1), 18 - x.size(2), device=x.device)
            x = torch.cat((x, padding), dim = 2)

        x = x.unsqueeze(1)

        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)

        x = self.shared_dense(x)

        out_string = self.head_string
        out_fret = self.head_fret
        out_pitch = self.head_pitch

        out_onset = torch.sigmoid(self.head_onset)
        out_offset = torch.sigmoid(self.head_offset)

        return out_string, out_fret, out_pitch, out_onset, out_offset