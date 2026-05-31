# datasets/dataset.py
import torch
from torch.utils.data import Dataset
from utils.signal_gen import *
from utils.stft import iq_to_spectrogram
import config


def build_dataset():
    X, y = [], []

    funcs = [gen_cw, gen_lfm, gen_bfsk, gen_bpsk, gen_qpsk]

    for label, func in enumerate(funcs):
        for _ in range(config.N_SAMPLES_PER_CLASS):
            sig = func()
            snr = np.random.randint(*config.SNR_RANGE)
            sig = add_noise(sig, snr)

            iq = to_iq(sig)
            spec = iq_to_spectrogram(iq)

            X.append(spec)
            y.append(label)

    X = np.array(X)[:, np.newaxis, :, :]
    y = np.array(y)

    return X, y


class RadioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]