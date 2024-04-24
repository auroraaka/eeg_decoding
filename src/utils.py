import torch
import numpy as np
import dataclasses
import typing as tp

from collections import defaultdict
from scipy.optimize import minimize
from transformers import AutoTokenizer
from torch.utils.data import Dataset

from src.config import Config
from src.datasets import BrennanHaleDataset, BroderickDataset
from src.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

from bm.events import Event
from bm.studies import Recording

def find_matching_electrodes():
    mod_config = Config("config/config.yaml")
    mod_config.datasets.brennan_hale.subjects = mod_config.datasets.broderick.subjects = ['S01']
    _, brennan_hale_positions = BrennanHaleDataset(mod_config).get_electrodes()
    _, broderick_positions = BroderickDataset(mod_config).get_electrodes()

    def transform_positions(r, positions):
        return {
            label: [r * np.sin(np.radians(theta)) * np.cos(np.radians(phi)),
                    r * np.sin(np.radians(theta)) * np.sin(np.radians(phi)),
                    r * np.cos(np.radians(theta))] 
            for label, (theta, phi) in positions.items()
        }

    def total_distance_error(r, positions1, positions2):
        positions2 = transform_positions(r[0], positions2)
        total_error = sum(
            min(np.linalg.norm(np.array(coord1) - np.array(coord2))
                for _, coord2 in positions2.items())
            for _, coord1 in positions1.items()
        )
        return total_error

    res = minimize(total_distance_error, x0=[10], args=(brennan_hale_positions, broderick_positions), bounds=[(1, 20)])
    r, error = res.x[0], res.fun 
    broderick_positions = transform_positions(r, broderick_positions)

    distance_mappings = defaultdict(list)
    for label1, coord1 in brennan_hale_positions.items():
        for label2, coord2 in broderick_positions.items():
            distance = np.linalg.norm(np.array(coord1) - np.array(coord2))
            distance_mappings[label1].append((label2, distance))

    for distances in distance_mappings.values():
        distances.sort(key=lambda x: x[1])

    closest_electrodes = {}
    assigned = set()
    for label1, distances in distance_mappings.items():
        for label2, _ in distances:
            if label2 not in assigned:
                closest_electrodes[label1] = label2
                assigned.add(label2)
                break

    return closest_electrodes, torch.tensor(list(closest_electrodes.values())) - 1

def prepare_inputs(config, eeg, subjects, labels):
    _, select_indices = find_matching_electrodes()
    _eeg = eeg[:, select_indices, :].float()
    _subjects = subjects[:]

    tokenizer = AutoTokenizer.from_pretrained(config.llama.model_name, token=config.llama.token)
    prompt = config.tokenizer.prompt

    labels = [" ".join(label) for label in labels]
    _label_ids = [tokenizer_image_token(prompt + DEFAULT_IMAGE_TOKEN + label, tokenizer, config.tokenizer.max_length, return_tensors=config.tokenizer.return_tensors) for label in labels]
    _input_ids = [tokenizer_image_token(prompt + DEFAULT_IMAGE_TOKEN, tokenizer, config.tokenizer.max_length, return_tensors=config.tokenizer.return_tensors) for i in range(len(_label_ids))]
    _input_ids = torch.cat(_input_ids, dim=0)
    _label_ids = torch.cat(_label_ids, dim=0)
    return _eeg, _subjects, _input_ids, _label_ids


def tokenizer_image_token(input, tokenizer, max_length, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    input_chunks = [tokenizer(chunk).input_ids for chunk in input.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(input_chunks) > 0 and len(input_chunks[0]) > 0 and input_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(input_chunks[0][0])
    for x in insert_separator(input_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors == 'pt':
        # return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        return torch.cat((torch.tensor(input_ids, dtype=torch.long).unsqueeze(0), torch.zeros((1, max_length - len(input_ids)), dtype=torch.long)), dim=1)
    return input_ids

@dataclasses.dataclass
class SegmentBatch:
    meg: torch.Tensor
    subject_index: torch.Tensor
    _recordings: tp.List[Recording] = dataclasses.field(default_factory=list)
    _event_lists: tp.List[tp.List[Event]] = dataclasses.field(default_factory=list)

    def __len__(self) -> int:
        return len(self.meg)

class EEGDataset(Dataset):

    def __init__(self, eegs, subjects, inputs, labels):
        self.eegs = eegs
        self.subjects = subjects
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg = self.eegs[idx]
        subject = self.subjects[idx]
        input_data = self.inputs[idx]
        label = self.labels[idx]
        return (eeg, subject, input_data), label


