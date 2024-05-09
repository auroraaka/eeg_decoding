import torch
import json
import os
import matplotlib.pyplot as plt
import numpy as np

from transformers import Trainer, TrainingArguments
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

def train(config, model, train_dataloader, valid_dataloader, optimizer, study_name):
    device = torch.device(config.train.device)
    model.to(device)
    train_losses = []
    train_perplexities = []
    valid_losses = []
    valid_perplexities = []
    
    plt.ion()
    epochs = config.train.epochs
    for epoch in range(epochs):

        model.train()
        total_train_loss = 0
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


        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for (eeg, subject, input_ids), labels in valid_dataloader:
                eeg, subject, input_ids, labels = eeg.to(device), subject.to(device), input_ids.to(device), labels.to(device)
                outputs = model(input_ids=input_ids, labels=labels, eegs=eeg, subject_index=subject)
                loss = outputs.loss
                total_valid_loss += loss.item()

                valid_loss = loss.item()
                valid_losses.append(valid_loss)
                valid_perplexity = torch.exp(torch.tensor(valid_loss))
                valid_perplexities.append(valid_perplexity.item())

        print(f"Epoch {epoch+1}, Train Loss: {total_train_loss/len(train_dataloader)}, Valid Loss: {total_valid_loss/len(valid_dataloader)}")
        save_epoch(config, train_losses, train_perplexities, valid_losses, valid_perplexities, epoch, study_name)
        update_plots(train_losses, train_perplexities, valid_losses, valid_perplexities)
    
    plt.ioff()
    data = {
        'train_losses': train_losses,
        'train_perplexities': train_perplexities,
        'valid_losses': valid_losses,
        'valid_perplexities': valid_perplexities
    }
    with open(f'{config.studies.path}/{study_name}/all_epochs.json', 'w') as file:
        json.dump(data, file)

    return train_losses, train_perplexities, valid_losses, valid_perplexities

def save_epoch(config, train_losses, train_perplexities, valid_losses, valid_perplexities, epoch, study_name):
    data = {
        'train_losses': train_losses,
        'train_perplexities': train_perplexities,
        'valid_losses': valid_losses,
        'valid_perplexities': valid_perplexities
    }
    save_folder = f'{config.studies.path}/{study_name}/epochs'
    save_path = f'{save_folder}/epoch_{epoch+1}.json'
    os.makedirs(save_folder, exist_ok=True)
    with open(save_path, 'w') as file:
        json.dump(data, file)

def update_plots(train_losses, train_perplexities, valid_losses, valid_perplexities):
    clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(train_losses[-100:], label='Train Loss', color='blue')
    ax1.plot(valid_losses[-100:], label='Valid Loss', color='green')
    ax1.set_title('Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(train_perplexities[-100:], label='Train Perplexity', color='red')
    ax2.plot(valid_perplexities[-100:], label='Valid Perplexity', color='orange')
    ax2.set_title('Perplexity')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Perplexity')
    ax2.legend()
    
    plt.pause(0.1)
    plt.show()

def train_with_trainer(config, model, train_dataloader, valid_dataloader):
    training_args = TrainingArguments(
        output_dir=f"{config.studies.path}/results",
        num_train_epochs=config.train.epochs,
        per_device_train_batch_size=config.train.batch_size,
        per_device_eval_batch_size=config.train.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{config.studies.path}/logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=valid_dataloader,
        compute_metrics=compute_metrics  # Define this function to compute metrics
    )

    trainer.train()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}