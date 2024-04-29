import os
import re
import mne
import numpy as np
import pandas as pd
import scipy.io
import torch
import torchaudio

from abc import ABC, abstractmethod
from scipy.signal import butter, lfilter, iirnotch
from itertools import chain


class EEGDataset(ABC):
    def __init__(
        self, path, subjects, original_fs, resampled_fs, interval, num_channels
    ):
        self.path = path
        self.EEG_PATH = self.path + "/EEG/"
        self.STIM_PATH = self.path + "/Stimuli/Text/"

        self.subjects = subjects if not None else self.detect_subjects()
        self.electrodes_cartesian, self.electrodes_spherical = self.get_electrodes()
        self.num_channels = (
            num_channels
            if num_channels <= len(self.electrodes_cartesian)
            else len(self.electrodes_cartesian)
        )

        self.original_fs = original_fs
        self.fs = resampled_fs
        self.resampler = torchaudio.transforms.Resample(original_fs, resampled_fs)
        self.interval = interval
        self.num_samples = int(self.interval * self.fs)

        self.data = {}
        self.labels = {}
        self.metadata = self.get_metadata()
        s = []
        for subject in self.subjects:
            print(f"Retrieving {subject}...")
            self.data[subject], self.labels[subject] = self.get_eeg(subject)
            s.append(subject)
        self.subjects = s
        self.num_subjects = len(self.subjects)
        self.get_words()

        self.data_shape = self.__getitem__(self.subjects[0])[0].shape
        self.labels_shape = len(self.__getitem__(self.subjects[0])[1])

    @abstractmethod
    def detect_subjects(self):
        pass

    @abstractmethod
    def get_electrodes(self):
        pass

    @abstractmethod
    def signal_process(self, x):
        pass

    @abstractmethod
    def get_eeg(self, subject="S01"):
        pass

    @abstractmethod
    def get_words(self):
        pass

    def __getitem__(self, subject):
        if subject not in self.subjects:
            raise ValueError(f"Subject {subject} not found.")
        return self.data[subject], self.labels[subject]

    def __len__(self):
        return len(self.subjects)

    def __str__(self):
        return f"""
┌─────────────────────────────┐
│ Dataset Information         │
└─────────────────────────────┘
Dataset: {self.path.split('/')[-2]}
Subjects: {', '.join(self.subjects)}
Number of subjects: {self.num_subjects}

┌─────────────────────────────┐
│ Stimuli Information         │
└─────────────────────────────┘
Type: Audio
Number of words: {self.num_words}

┌─────────────────────────────┐
│ Signal Information          │
└─────────────────────────────┘
Number of channels: {self.num_channels}
Original sampling frequency: {self.original_fs} Hz
Resampled sampling frequency: {self.fs} Hz
Interval: {self.interval} seconds
Number of samples: {self.num_samples}

┌─────────────────────────────┐
│ Shape Information           │
└─────────────────────────────┘
Single Subject Data Shape: {self.data_shape}
Single Subject Labels Shape: {self.labels_shape}

"""


def spherical_to_cartesian(theta, phi, r=9):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return float(x), float(y), float(z)

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.rad2deg(np.arccos(z / r))
    phi = np.rad2deg(np.arctan2(y, x))
    theta, phi = round(theta), round(phi)

    if phi >= 90:
        theta, phi = -theta, -180+phi
    if phi <= -90:
        theta, phi = -theta, 180+phi
    if (theta < 0 and abs(phi-90) < 1):
        theta, phi = -theta, -phi
    return theta, phi

class BrennanHaleDataset(EEGDataset):
    def __init__(self, config):
        path = config.datasets.path + "/" + config.datasets.brennan_hale.path
        subjects = config.datasets.brennan_hale.subjects
        original_fs = config.datasets.brennan_hale.original_fs
        num_channels = config.datasets.brennan_hale.num_channels

        resampled_fs = config.datasets.resampled_fs
        interval = config.datasets.interval

        super().__init__(
            path, subjects, original_fs, resampled_fs, interval, num_channels
        )

    def detect_subjects(self):
        return {
            file[:3] for file in os.listdir(self.EEG_PATH) if re.search(r"S\d{2}", file)
        }

    def get_electrodes(self):
        if os.path.exists(self.path + '/easycap_M10.txt'):
            with open(self.path + '/easycap_M10.txt', 'r') as file:
                next(file)
                electrodes_spherical = {line.strip().split()[0]: (float(line.strip().split()[1]), float(line.strip().split()[2])) for line in file}
            electrodes_cartesian = {site: spherical_to_cartesian(*angles) for site, angles in electrodes_spherical.items()}
            return electrodes_cartesian, electrodes_spherical
        else:
            return self.convert_electrodes()

    def convert_electrodes(self):
        
        electrodes_cartesian = {}
        with open(self.path + "/easycapM10-acti61_elec.sfp", "r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 4:
                    site, x, y, z = parts
                    coords = [float(x), float(y), float(z)]
                    electrodes_cartesian[site] = coords
                    
        electrodes_spherical = {site: cartesian_to_spherical(*coords) for site, coords in electrodes_cartesian.items()}
        with open(self.path + '/easycap_M10.txt', 'w') as file:
            file.write("Site\tTheta\tPhi\n")
            for site, angles in electrodes_spherical.items():
                theta, phi = angles
                file.write(f"{site}\t{theta}\t{phi}\n")

        return electrodes_cartesian, electrodes_spherical


    def signal_process(self, raw):
        # High Pass Filter 0.1 Hz
        raw.filter(l_freq=0.1, h_freq=None, verbose=False)

        # Notch Filter 60 Hz
        raw.notch_filter(freqs=60, verbose=False)

    def get_eeg(self, subject="S01"):
        raw = mne.io.read_raw_brainvision(
            self.EEG_PATH + f"{subject}.vhdr", preload=True, verbose=False
        )
        self.signal_process(raw)
        data = raw.get_data(verbose=False)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        events_idx = events[:, 0].tolist()
        if len(events_idx) > 12:
            events_idx = events_idx[1:]

        eeg_samples = []
        word_samples = []
        for segment, words_times in self.metadata.items():
            for words, times in zip(*words_times):
                word_samples.append(words)
                start_t, end_t = times
                start_idx = int(start_t * self.original_fs + events_idx[segment])
                end_idx = int(end_t * self.original_fs + events_idx[segment])
                eeg_samples.append(data[: self.num_channels, start_idx:end_idx])

        eeg_samples = self.resampler(
            torch.tensor(np.array(eeg_samples), dtype=torch.float)
        )
        return eeg_samples, word_samples

    def get_metadata(self):
        self.df = pd.read_csv(self.path + "/AliceChapterOne-EEG.csv")
        grouped = self.df.groupby("Segment")
        metadata = {}
        for segment, group in grouped:
            words = []
            times = []
            current_words = []
            start = 0.0
            end = self.interval
            for word, onset, offset in zip(
                group["Word"].tolist(),
                group["onset"].tolist(),
                group["offset"].tolist(),
            ):
                if onset >= end:
                    words.append(current_words)
                    times.append((start, end))

                    current_words = [word]
                    start = end
                    end += self.interval
                else:
                    current_words.append(word)
            if current_words:
                words.append(current_words)
                times.append((start, end))
            metadata[segment - 1] = (words, times)
        return metadata

    def get_words(self):
        self.order = self.df["Order"].tolist()
        self.words = self.df["Word"].tolist()
        self.num_words = len(self.words)
        self.word2order = {w: o - 1 for o, w in zip(self.order, self.words)}
        self.order2word = {o - 1: w for o, w in zip(self.order, self.words)}


class BroderickDataset(EEGDataset):
    def __init__(self, config):
        path = config.datasets.path + "/" + config.datasets.broderick.path
        subjects = config.datasets.broderick.subjects
        original_fs = config.datasets.broderick.original_fs
        num_channels = config.datasets.broderick.num_channels
        
        resampled_fs = config.datasets.resampled_fs
        interval = config.datasets.interval
        
        super().__init__(
            path, subjects, original_fs, resampled_fs, interval, num_channels
        )

    def detect_subjects(self):
        return {
            "S" + file.replace("Subject", "").zfill(2)
            for file in os.listdir(self.EEG_PATH)
            if "Subject" in file
        }

    def get_electrodes(self):
        electrodes_spherical = {}
        with open(self.EEG_PATH + "biosemi128.txt", "r") as file:
            next(file)
            i = 0
            for line in file:
                i += 1
                if i > 128:
                    break
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    site, theta, phi = parts
                    electrodes_spherical[i] = [float(theta), float(phi)]
        electrodes_cartesian = {site: spherical_to_cartesian(*angles) for site, angles in electrodes_spherical.items()}
        return electrodes_cartesian, electrodes_spherical

    def signal_process(self, x):
        def highpass_filter(x, cutoff=0.1, order=5):
            nyq = 0.5 * self.fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype="high", analog=False)
            y = lfilter(b, a, x)
            return y

        def notch_filter(x, freq=60, quality=30):
            b, a = iirnotch(freq, quality, self.fs)
            y = lfilter(b, a, x)
            return y

        y = highpass_filter(x)
        y = notch_filter(y)
        return y

    def get_eeg(self, subject="S01"):
        subject_path = (
            self.EEG_PATH + "Subject" + str(int(subject.replace("S", ""))) + "/"
        )
        resp_dir = sorted(
            [
                subject_path + file
                for file in os.listdir(subject_path)
                if ".mat" in file
            ],
            key=self.sort_key,
        )
        eeg_samples = []
        word_samples = []
        for resp_path, words_times in zip(resp_dir, self.metadata.values()):
            data = scipy.io.loadmat(resp_path)["eegData"]
            data = data.reshape(self.num_channels, -1)
            data = self.signal_process(data)
            for words, times in zip(*words_times):
                word_samples.append(words)
                start_t, end_t = times
                start_idx = int(start_t * self.original_fs)
                end_idx = int(end_t * self.original_fs)
                eeg_samples.append(data[: self.num_channels, start_idx:end_idx])
        eeg_samples = self.pad_eeg(eeg_samples)
        eeg_samples = self.resampler(
            torch.tensor(np.array(eeg_samples), dtype=torch.float)
        )
        return eeg_samples, word_samples

    def pad_eeg(self, eeg_samples):
        padded = []
        target_shape = (self.num_channels, self.original_fs * self.interval)
        for eeg in eeg_samples:
            padding = [(0, max(target_shape[i] - eeg.shape[i], 0)) for i in range(2)]
            padded_eeg = np.pad(
                eeg, pad_width=padding, mode="constant", constant_values=0
            )
            padded.append(padded_eeg)
        return padded

    def get_metadata(self):
        runs = self.get_runs()
        metadata = {}
        for run, stim in runs.items():
            words = []
            times = []
            current_words = []
            start = 0.0
            end = self.interval
            for word, onset, offset in zip(*stim):
                if onset >= end:
                    words.append(current_words)
                    times.append((start, end))

                    current_words = [word]
                    start = end
                    end += self.interval
                else:
                    current_words.append(word)
            if current_words:
                words.append(current_words)
                times.append((start, end))
            metadata[run] = (words, times)
        return metadata

    def sort_key(self, filename):
        numbers = re.findall(r"\d+", filename)
        return int(numbers[0]) if numbers else 0

    def get_runs(self):
        stim_dir = sorted(
            [
                self.STIM_PATH + file
                for file in os.listdir(self.STIM_PATH)
                if ".mat" in file
            ],
            key=self.sort_key,
        )
        runs = {}
        for i, stim_file in enumerate(stim_dir):
            runs[i] = (
                [word[0][0].lower() for word in scipy.io.loadmat(stim_file)["wordVec"]],
                scipy.io.loadmat(stim_file)["onset_time"],
                scipy.io.loadmat(stim_file)["offset_time"],
            )
        return runs

    def get_words(self):
        self.words = list(
            chain.from_iterable([run[0] for run in self.get_runs().values()])
        )
        self.order = list(range(len(self.words)))
        self.num_words = len(self.words)
        self.word2order = {w: o - 1 for o, w in zip(self.order, self.words)}
        self.order2word = {o - 1: w for o, w in zip(self.order, self.words)}