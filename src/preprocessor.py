import torch
import numpy as np

from scipy import signal, stats
from sklearn.preprocessing import RobustScaler

class Preprocessor:
    def __init__(self, config, EEG):
        self.EEG = EEG
        self.baseline_length = config.preprocessor.baseline_length
        self.stft_flag = config.preprocessor.stft_flag
        self.nperseg = config.preprocessor.nperseg
        self.noverlap = config.preprocessor.noverlap
        self.normalizing = config.preprocessor.normalizing
        self.return_onesided = config.preprocessor.return_onesided
        self.num_frequencies = config.preprocessor.nperseg / 2 + 1

        self.preprocessed = {}
        for subject, data in self.EEG.data.items():
            print(f"Processing {subject}...")
            data = self.get_baseline_average(data)
            data = self.get_scaled(data)
            data = self.get_clamped(data)
            if self.stft_flag:
                data = self.get_stft(data)
            self.preprocessed[subject] = data

    def get_baseline_average(self, x):
        baseline_average = x[:, :, : int(self.EEG.fs * self.baseline_length)].mean(
            axis=2, keepdim=True
        )
        x -= baseline_average
        return x

    def get_scaled(self, x):
        x_flat = x.reshape(-1, x.shape[-1])
        scaler = RobustScaler()
        x_scaled = scaler.fit_transform(x_flat)
        x_scaled = torch.tensor(x_scaled.reshape(x.shape))
        return x_scaled

    def get_clamped(self, x):
        std_dev = x.std(dim=2, keepdim=True)
        x_clamped = torch.where(x > 20 * std_dev, 20 * std_dev, x)
        return x_clamped

    def normalize_stft(self, x, **kwargs):
        f, t, Zxx = signal.stft(x, **kwargs)
        # Zxx = Zxx[:self.clip_fs]
        # f = f[:self.clip_fs]
        Zxx = np.abs(Zxx)
        # clip = 5
        if self.normalizing == "zscore":
            # Zxx = Zxx[:, clip:-clip]
            Zxx = stats.zscore(Zxx, axis=-1)
            # t = t[clip:-clip]
        # elif self.normalizing == "db":
        #     Zxx = np.log2(Zxx[:, clip:-clip])
        #     t = t[clip:-clip]

        if np.isnan(Zxx).any():
            import pdb

            pdb.set_trace()

        return f, t, Zxx

    def get_stft(self, eeg):
        stft_eeg = []
        for i in range(self.EEG.labels_shape):
            stft_channels = []
            for j in range(self.EEG.num_channels):
                f, t, stft = self.normalize_stft(
                    eeg[i, j],
                    fs=self.EEG.fs,
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                    return_onesided=self.return_onesided,
                )
                stft_channels.append(stft)
            stft_eeg.append(stft_channels)

        return torch.tensor(np.array(stft_eeg))

    def __getitem__(self, subject):
        if subject == "ALL":
            return (
                torch.cat(tuple(self.preprocessed.values()), 0),
                torch.cat(
                    [
                        torch.ones((self.EEG.labels_shape), dtype=torch.int64) * int(subject[1:])
                        for subject in self.preprocessed.keys()
                    ],
                    0,
                ),
                self.EEG.labels[self.EEG.subjects[0]] * self.EEG.num_subjects,
            )
        else:
            if subject in self.preprocessed:
                return (
                    self.preprocessed[subject],
                    torch.ones((self.EEG.labels_shape), dtype=torch.int64) * int(subject[1:]),
                    self.EEG.labels[self.EEG.subjects[0]],
                )
            else:
                raise ValueError(f"Subject {subject} not found.")