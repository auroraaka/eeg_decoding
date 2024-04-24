from src.config import Config
from src.datasets import BroderickDataset
from src.preprocessor import Preprocessor
from src.utils import prepare_inputs, EEGDataset
from src.model import EEGAdapterLlamaForCausalLM

from torch.utils.data import DataLoader
import torch

if __name__ == "__main__":
    
    config = Config("config/config.yaml")
    EEG = BroderickDataset(config)
    PROCESSOR = Preprocessor(config, EEG=EEG)
    eegs, subjects, inputs, labels = prepare_inputs(config, *PROCESSOR['ALL'])
    braindecoder = EEGAdapterLlamaForCausalLM(config, config.llama.model_name, config.llama.token)

    dataset = EEGDataset(eegs, subjects, inputs, labels)
    dataloader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True)
    test_eegs = eegs[:2, :, :].float()
    test_subjects = subjects[:2]
    test_inputs = torch.cat(inputs[:2], dim=0)
    test_labels = torch.cat(labels[:2], dim=0)
    
    output = braindecoder(input_ids=test_inputs, labels=test_labels, eegs=test_eegs, subject_index=test_subjects)
    predicted_token_ids = torch.argmax(output.logits, dim=-1)

