from src.config import Config
from src.datasets import BroderickDataset
from src.preprocessor import Preprocessor
from src.utils import prepare_inputs, EEGDataset
from src.model import EEGAdapterLlamaForCausalLM

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

if __name__ == "__main__":
    
    config = Config("config/config.yaml")
    EEG = BroderickDataset(config)
    PROCESSOR = Preprocessor(config, EEG=EEG)
    eegs, subjects, inputs, labels = prepare_inputs(config, *PROCESSOR['ALL'])
    
    # braindecoder = EEGAdapterLlamaForCausalLM(config, config.llama.model_name, config.llama.token)
    # for name, param in braindecoder.named_parameters():
    #     if param.requires_grad:
    #         print(f"Parameter: {name}, Size: {param.size()}")
    #     else:
    #         print(f"Frozen Parameter: {name}, Size: {param.size()}")

    # eegs, subjects, inputs, labels = eegs.to('cuda'), torch.tensor(subjects).to('cuda'), inputs.to('cuda'), labels.to('cuda')
    # dataset = EEGDataset(eegs, subjects, inputs, labels)
    # dataloader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True)
    # optimizer = Adam(braindecoder.parameters(), lr=config.train.learning_rate)
    
    # def train(model, dataloader, optimizer, epochs, device):

    #     model.train()
    #     model.to(device)
    #     for epoch in range(epochs):
    #         total_loss = 0
    #         for (eeg, subject, input_ids), labels in dataloader:

    #             optimizer.zero_grad()

    #             outputs = model(input_ids=input_ids, labels=labels, eegs=eeg.float(), subject_index=subject)
    #             loss = outputs.loss
    #             total_loss += loss.item()
    #             print(loss.item())

    #             loss.backward()

    #             optimizer.step()

    #         avg_loss = total_loss / len(dataloader)
    #         print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    # device = torch.device("cuda")
    # train(braindecoder, dataloader, optimizer, config.train.epochs, device)
    
    # output = braindecoder(input_ids=test_inputs, labels=test_labels, eegs=test_eegs, subject_index=test_subjects)
    # predicted_token_ids = torch.argmax(output.logits, dim=-1)

