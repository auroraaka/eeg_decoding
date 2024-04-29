from src.config import Config
from src.datasets import BroderickDataset
from src.preprocessor import Preprocessor
from src.utils import prepare_inputs, DatasetWrapper
from src.model import EEGAdapterLlamaForCausalLM
from src.train import train, make_splits

import torch
from torch.optim import Adam

if __name__ == "__main__":
    
    config = Config("config/config.yaml")
    EEG = BroderickDataset(config)
    PROCESSOR = Preprocessor(config, EEG=EEG)
    eegs, subjects, inputs, labels = prepare_inputs(config, *PROCESSOR['ALL'])

    # braindecoder = EEGAdapterLlamaForCausalLM(config, config.llama.model_name, config.llama.token)
    # dataset = DatasetWrapper(eegs, subjects, inputs, labels)
    # train_loader, val_loader, test_loader = make_splits(config, dataset)
    # optimizer = Adam(braindecoder.parameters(), lr=config.train.learning_rate)
    # train(braindecoder, train_loader, val_loader, optimizer, config.train.epochs, torch.device("cpu"))