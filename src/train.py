import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from IPython.display import clear_output

def make_splits(config, dataset, train=0.8, val=0.1, test=0.1):
    train_size = int(len(dataset) * train)
    val_size = int(len(dataset) * val)
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader =  DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True)
    valid_loader = DataLoader(val_set, batch_size=config.train.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config.train.batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader

def train(model, train_dataloader, valid_dataloader, optimizer, epochs, device):
    model.to(device)
    train_losses = []
    train_perplexities = []
    valid_losses = []
    valid_perplexities = []
    
    plt.ion()
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        iteration = 0  

        for (eeg, subject, input_ids), labels in train_dataloader:
            eeg, subject, input_ids, labels = eeg.to(device), subject.to(device), input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels, eegs=eeg, subject_index=subject)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

            curr_loss = loss.item()
            train_losses.append(curr_loss)
            train_perplexity = torch.exp(torch.tensor(curr_loss))
            train_perplexities.append(train_perplexity.item())

            iteration += 1
            if iteration % 5 == 0:
                model.eval()
                total_valid_loss = 0
                for (eeg, subject, input_ids), labels in valid_dataloader:
                    eeg, subject, input_ids, labels = eeg.to(device), subject.to(device), input_ids.to(device), labels.to(device)
                    outputs = model(input_ids=input_ids, labels=labels, eegs=eeg, subject_index=subject)
                    valid_loss = outputs.loss.item()
                    total_valid_loss += valid_loss
                    valid_losses.append(valid_loss)
                    valid_perplexity = torch.exp(torch.tensor(valid_loss))
                    valid_perplexities.append(valid_perplexity.item())

                update_plots(train_losses, train_perplexities, valid_losses, valid_perplexities)

        print(f"Epoch {epoch+1}, Train Loss: {total_train_loss/len(train_dataloader)}, Valid Loss: {total_valid_loss/len(valid_dataloader)}")
    
    plt.ioff()


def update_plots(train_losses, train_perplexities, valid_losses, valid_perplexities):
    clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(train_losses[-30:], label='Train Loss', color='blue')
    ax1.plot(valid_losses[-30:], label='Valid Loss', color='green')
    ax1.set_title('Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(train_perplexities[-30:], label='Train Perplexity', color='red')
    ax2.plot(valid_perplexities[-30:], label='Valid Perplexity', color='orange')
    ax2.set_title('Perplexity')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Perplexity')
    ax2.legend()
    
    plt.pause(0.1)
    plt.show()
