
import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
import argparse
from tqdm import tqdm

from models.utils import *
from models.transformer import Transformer
from shuffles.shuffles import *


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=20, metavar='N')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)

    # Sequence generation hyperparameters
    parser.add_argument('--shuffle_type', type=str, default='CutOnlyShuffle')
    parser.add_argument('--num_decks', type=int, default=1)
    parser.add_argument('--shuffle_threshold', type=float, default=0.5)
    parser.add_argument('--sequence_length', type=int, default=200)
    parser.add_argument('--num_train_sequences', type=int, default=10000)
    parser.add_argument('--num_valid_sequences', type=int, default=1000)

    args = parser.parse_args()
    return args


def create_shuffle(args):
    if args.shuffle_type == 'CutOnlyShuffle':
        return CutOnlyShuffle(args.num_decks, args.shuffle_threshold)
    elif args.shuffle_type == 'HomeShuffle':
        return HomeShuffle(args.num_decks, args.shuffle_threshold)
    elif args.shuffle_type == 'CasinoHandShuffle':
        return CasinoHandShuffle(args.num_decks, args.shuffle_threshold)
    elif args.shuffle_type == 'AutomaticMachineShuffle':
        return AutomaticMachineShuffle(args.num_decks, args.shuffle_threshold)
    else:
        raise Exception('Unknown shuffle type')

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(vocab_size=len(card_to_int), num_positions=200, d_model=4, num_heads=2, dim_feedforward=8, num_classes=2, num_layers=2).to(device)

    loss_fn = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    shuffle = create_shuffle(args)
    train_seqs = [shuffle.generate_deal_sequence(args.sequence_length) for _ in range(args.num_train_sequences)]
    valid_seqs = [shuffle.generate_deal_sequence(args.sequence_length) for _ in range(args.num_valid_sequences)]
    train_dataloader = DataLoader(CardDataset(train_seqs), batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(CardDataset(valid_seqs), batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0
        random.seed(epoch)
        model.train()
        
        for inputs, targets in tqdm(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, causal_mask(inputs.shape[1]))
            loss = loss_fn(outputs, targets)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        model.eval()
        valid_loss = 0
        for inputs, targets in valid_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, causal_mask(inputs.shape[1]))
            loss = loss_fn(outputs, targets)
            valid_loss += loss.item()
        print('Epoch %d: avg loss = %.4f, avg valid loss = %.4f, time = %.2fs' % (epoch, epoch_loss / len(train_dataloader), valid_loss / len(valid_dataloader), time.time() - epoch_start_time))


if __name__ == '__main__':
    args = parse_args()
    train(args)