import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch

def update_plots(losses, perplexities):
    clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.plot(losses, label='Loss', color='blue')
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(perplexities, label='Perplexity', color='red')
    ax2.set_title('Perplexity over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.legend()
    
    plt.pause(0.1)
    plt.show()

def train(model, dataloader, optimizer, epochs, device):
    model.train()
    model.to(device)
    losses = []
    perplexities = []
    
    plt.ion()
    
    for epoch in range(epochs):
        total_loss = 0
        for (eeg, subject, input_ids), labels in dataloader:
            eeg, subject, input_ids, labels = eeg.to(device), subject.to(device), input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels, eegs=eeg, subject_index=subject)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        perplexity = torch.exp(torch.tensor(avg_loss))
        perplexities.append(perplexity.item())
        
        update_plots(losses, perplexities)
    
    plt.ioff()

