from src.config import Config
from src.datasets import BroderickDataset
from src.preprocessor import Preprocessor
from src.utils import prepare_inputs, EEGDataset
from src.model import EEGAdapterLlamaForCausalLM
from src.train import train

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

if __name__ == "__main__":
    
    config = Config("config/config.yaml")
    EEG = BroderickDataset(config)
    PROCESSOR = Preprocessor(config, EEG=EEG)
    eegs, subjects, inputs, labels = prepare_inputs(config, *PROCESSOR['ALL'])

    braindecoder = EEGAdapterLlamaForCausalLM(config, config.llama.model_name, config.llama.token)
    dataset = EEGDataset(eegs, subjects, inputs, labels)

    def make_splits(dataset, train=0.8, val=0.1, test=0.1):
        train_size = int(len(dataset) * train)
        val_size = int(len(dataset) * val)
        test_size = int(len(dataset) * test)

        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

        train_loader =  DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True)
        valid_loader = DataLoader(val_set, batch_size=config.train.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=config.train.batch_size, shuffle=False)
        
        return train_loader, valid_loader, test_loader
    
    train_loader, val_loader, test_loader = make_splits(dataset)
    optimizer = Adam(braindecoder.parameters(), lr=config.train.learning_rate)
    train(braindecoder, train_loader, val_loader, test_loader, optimizer, config.train.epochs, torch.device(config.train.device))

