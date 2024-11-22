import os
import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List, Union
import argparse
from tqdm import tqdm
import pandas as pd

from models.utils import *
from models.transformer import Transformer
from shuffles.shuffles import *


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')

    # Model architecture hyperparameters
    parser.add_argument('--d_model', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=2)

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=20, metavar='N')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)

    # Sequence generation hyperparameters
    parser.add_argument('--shuffle_type', type=str, default='RiffleOnlyShuffle')
    parser.add_argument('--num_riffle_passes', type=int, default=1)
    parser.add_argument('--num_decks', type=int, default=1)
    parser.add_argument('--shuffle_threshold', type=float, default=0.5)
    parser.add_argument('--sequence_length', type=int, default=200)
    parser.add_argument('--num_train_sequences', type=int, default=10000)
    parser.add_argument('--num_valid_sequences', type=int, default=1000)

    # Output
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--output_df', type=bool)

    args = parser.parse_args()
    return args


# Function to provide default arguments for external calls
def get_default_args():
    return {
        'd_model': 4,
        'num_heads': 2,
        'dim_feedforward': 8,
        'num_layers': 2,
        'batch_size': 20,
        'num_epochs': 10,
        'learning_rate': 0.01,
        'shuffle_type': 'RiffleOnlyShuffle',
        'num_riffle_passes': 1,
        'num_decks': 1,
        'shuffle_threshold': 0.5,
        'sequence_length': 200,
        'num_train_sequences': 10000,
        'num_valid_sequences': 1000,
        'output_file': None,
        'output_df': False,
    }


def create_shuffle(args):
    if args.shuffle_type == 'RiffleOnlyShuffle':
        return RiffleOnlyShuffle(args.num_riffle_passes, args.num_decks, args.shuffle_threshold)
    elif args.shuffle_type == 'CutOnlyShuffle':
        return CutOnlyShuffle(args.num_decks, args.shuffle_threshold)
    elif args.shuffle_type == 'HomeShuffle':
        return HomeShuffle(args.num_decks, args.shuffle_threshold)
    elif args.shuffle_type == 'CasinoHandShuffle':
        return CasinoHandShuffle(args.num_decks, args.shuffle_threshold)
    elif args.shuffle_type == 'AutomaticMachineShuffle':
        return AutomaticMachineShuffle(args.num_decks, args.shuffle_threshold)
    else:
        raise Exception('Unknown shuffle type')


def train(args: Union[argparse.Namespace, dict]):
    # If args is a dictionary, convert to Namespace
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    
    output_name = args.output_file
    if not output_name:
        raise ValueError("Output name must be specified")

    # Directory to save attention maps
    attn_maps_dir = os.path.join('attn_maps', f"{output_name}_attn_maps")
    os.makedirs(attn_maps_dir, exist_ok=True)

    # File for saving epoch results
    epoch_output_file = os.path.join('experiment_data', f"{output_name}_epoch_results.csv")
    # Initialize the CSV file with headers
    with open(epoch_output_file, 'w') as f:
        f.write("epoch,avg_train_loss,avg_valid_loss,valid_accuracy,precision,recall,elapsed_time\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(
        vocab_size=len(card_to_int), 
        num_positions=200, 
        d_model=args.d_model, 
        num_heads=args.num_heads, 
        dim_feedforward=args.dim_feedforward, 
        num_classes=2, 
        num_layers=args.num_layers,
    ).to(device)

    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    shuffle = create_shuffle(args)
    train_seqs = [shuffle.generate_deal_sequence(args.sequence_length) for _ in range(args.num_train_sequences)]
    valid_seqs = [shuffle.generate_deal_sequence(args.sequence_length) for _ in range(args.num_valid_sequences)]
    train_dataloader = DataLoader(CardDataset(train_seqs), batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(CardDataset(valid_seqs), batch_size=args.batch_size, shuffle=False)  # Fixed order for consistent validation

    # Select fixed validation examples
    fixed_validation_indices = random.sample(range(len(valid_seqs)), 1)
    fixed_validation_examples = [valid_seqs[i] for i in fixed_validation_indices]
    fixed_validation_dataloader = DataLoader(CardDataset(fixed_validation_examples), batch_size=1, shuffle=False)

    best_valid_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0
        random.seed(epoch)
        model.train()
        
        for inputs, targets in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{args.num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs, causal_mask(inputs.shape[1]))
            loss = loss_fn(outputs, targets)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        model.eval()
        valid_loss = 0
        correct_predictions = 0
        total_predictions = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        with torch.no_grad():
            for inputs, targets in valid_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = model(inputs, causal_mask(inputs.shape[1]))
                loss = loss_fn(outputs, targets)
                valid_loss += loss.item()

                # Calculate predictions and accuracy
                probs = torch.exp(outputs)  # Convert log probabilities to probabilities
                preds = torch.argmax(probs, dim=1)  # Predicted class (0 or 1)
                
                # Update metrics
                correct_predictions += (preds == targets).sum().item()
                total_predictions += targets.size(0)
                true_positives += ((preds == 1) & (targets == 1)).sum().item()
                false_positives += ((preds == 1) & (targets == 0)).sum().item()
                false_negatives += ((preds == 0) & (targets == 1)).sum().item()

        avg_train_loss = epoch_loss / len(train_dataloader)
        avg_valid_loss = valid_loss / len(valid_dataloader)
        validation_accuracy = correct_predictions / total_predictions
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        elapsed_time = time.time() - epoch_start_time

        scheduler.step(avg_valid_loss)

        # Append epoch results to the CSV file
        with open(epoch_output_file, 'a') as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{avg_valid_loss:.4f},{validation_accuracy:.4f},{precision:.4f},{recall:.4f},{elapsed_time:.2f}\n")

        # Save attention maps and inputs for fixed validation examples
        with torch.no_grad():
            for example_idx, (inputs, targets) in enumerate(fixed_validation_dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                _, attn_maps = model(inputs, causal_mask(inputs.shape[1]))

                # Combine inputs and attention maps into a dictionary
                save_data = {
                    'inputs': inputs.cpu(),  # Save inputs as a CPU tensor
                    'attention_maps': attn_maps,  # Save attention maps as a CPU tensor
                }

                # Save the dictionary to a .pt file
                save_file = os.path.join(attn_maps_dir, f"epoch_{epoch}_example_{example_idx}_data.pt")
                torch.save(save_data, save_file)

        print(f"Epoch {epoch}: avg_train_loss = {avg_train_loss:.4f}, avg_valid_loss = {avg_valid_loss:.4f}, time = {elapsed_time:.2f}s")
        
        # Early stopping
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join('checkpoints', f"{output_name}_best_model.pth"))
            print(f"Validation loss improved. Saving model...")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break




if __name__ == '__main__':
    args = parse_args()
    train(args)