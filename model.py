import torch
import torch.nn as nn
import nnAudio.features

SR = 22050
HOP_LENGTH = 256

CQT_BINS = 630
INPUT_FRAMES = 18

TAB_CLASSES = (1 + 22) * 4 #(1 OPEN + 22 FRETS) * 4 STRINGS
PITCH_BINS = CQT_BINS
TECHNIQUE_CLASS = 1 #finger-style

class BassTranscriptionCNN(nn.Module):
    def __init__(self):
        super(BassTranscriptionCNN, self).__init__()

        self.cqt = nnAudio.features.CQT(
            sr = SR,
            hop_length = HOP_LENGTH,
            fmin = 41.2, #E1
            n_bins = CQT_BINS,
            bins_per_octave = 120
        )

        self.conv_body = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(3,3),
                padding="valid"
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3,3),
                padding="valid"
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3,3),
                padding="valid"
            ),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=(2,2),
                stride=(2,2),
                padding="valid"
            ),

            nn.Flatten()
        )

        self.shared_dense = nn.Sequential(
            nn.Linear(312 * 6 * 64, 128),
            nn.ReLU()
        )

