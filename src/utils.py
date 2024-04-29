import torch
import json
import os
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

def find_matching_electrodes(config):
    mod_config = config.deepcopy()
    mod_config.datasets.brennan_hale.subjects = mod_config.datasets.broderick.subjects = ['S01']
    bh = BrennanHaleDataset(mod_config).electrodes
    br = BroderickDataset(mod_config).electrodes

    bh2br = {}
    br = np.array(list(br.values()))
    for bh_site, (theta, phi) in bh.items():
        distances = np.linalg.norm(br - np.array([theta, phi]), axis=1)
        distances[list(bh2br.values())] = np.inf
        br_site = np.argmin(distances)
        bh2br[int(bh_site)-1] = int(br_site)

    with open(config.datasets.path + '/electrode_mapping.json', 'w') as file:
        json.dump(bh2br, file)

    return bh2br

def get_indices(site2site):
    indices = [0] * len(site2site)
    for s1, s2 in site2site.items():
        indices[int(s1)] = int(s2)
    return indices
    
def retrieve_matching_electrodes(config):
    path = config.datasets.path + '/electrode_mapping.json'
    if os.path.exists(path):
        with open(path, 'r') as file:
            bh2br = json.load(file)
    else:
        bh2br = find_matching_electrodes(config)
    return get_indices(bh2br)

def prepare_inputs(config, eeg, subjects, labels):
    electrode_indices = retrieve_matching_electrodes(config)
    _eeg = eeg[:, electrode_indices, :].float()
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

class DatasetWrapper(Dataset):

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


