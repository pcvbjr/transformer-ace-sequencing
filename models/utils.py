import torch
from torch.utils.data import Dataset, DataLoader

# Step 1: Map each card to an integer
card_to_int = {
    'shuffle': 0,
    '2H': 1, '3H': 2, '4H': 3, '5H': 4, '6H': 5, '7H': 6, '8H': 7, '9H': 8, '10H': 9, 'JH': 10, 'QH': 11, 'KH': 12, 'AH': 13,
    '2D': 14, '3D': 15, '4D': 16, '5D': 17, '6D': 18, '7D': 19, '8D': 20, '9D': 21, '10D': 22, 'JD': 23, 'QD': 24, 'KD': 25, 'AD': 26,
    '2S': 27, '3S': 28, '4S': 29, '5S': 30, '6S': 31, '7S': 32, '8S': 33, '9S': 34, '10S': 35, 'JS': 36, 'QS': 37, 'KS': 38, 'AS': 39,
    '2C': 40, '3C': 41, '4C': 42, '5C': 43, '6C': 44, '7C': 45, '8C': 46, '9C': 47, '10C': 48, 'JC': 49, 'QC': 50, 'KC': 51, 'AC': 52,
}

# Step 2: Define your dataset
class CardDataset(Dataset):
    def __init__(self, sequences):
        self.data = []

        for seq in sequences:
            input_seq = [card_to_int[card] for card in seq]
            target_seq = [0] * len(input_seq)
            for j in range(len(input_seq) - 2):
                if input_seq[j+1] % 13 == 0:
                    target_seq[j] = 1
            self.data.append((input_seq, target_seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])


def causal_mask(chunk_size):
    # Create a square matrix of ones of size (chunk_size, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size), diagonal=1)
    
    # Replace 1s with negative infinity and 0s with 0s
    mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, 0)
    
    return mask