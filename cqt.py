import nnAudio.features

SR = 22050
HOP_LENGTH = 256

CQT_BINS = 630
INPUT_FRAMES = 18

cqt = nnAudio.features.CQT(
    sr = SR,
    hop_length = HOP_LENGTH,
    fmin = 32.7, #E1
    n_bins = CQT_BINS,
    bins_per_octave = 120
)