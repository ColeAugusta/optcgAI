import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Card import Card
from Decklist import Decklist

# convert decklists -> dataset
class DeckDataset(Dataset):
    def __init__(self, decklist: Decklist):
        self.data = []
        self.data.append(self.cards_to_tensor(decklist))
        self.data.append(decklist.winrate)

    def __len__(self):
        return len(self.data)

    # build tensors form decklist cards
    def cards_to_tensor(self, decklist: Decklist) -> torch.Tensor:
        features = []
        for card in decklist.cards:
            feature_vec = [
                # simple hashing for tensors
                hash(card.number) % 10000,
                hash(card.card_type) % 1000,
                hash(card.name) % 100000,
                card.appearance_rate,
                card.decks_appeared
            ] + card.number_appeared

            features.append(feature_vec)
        
        return torch.tensor(features, dtype=torch.float32)

# Nueral network optimizer for single leader
class DeckOptimizer(nn.Module):
    def __init__(self, card_feature_dim=9, hidden_dim=128):
        super().__init__()

        self.card_encoder = nn.Sequential(
            nn.Linear(card_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

if __name__ == "__main__":
    bonney = Decklist("../data/op12BonneyCards.csv", 0.67)
    dataset = DeckDataset(bonney)
    dataloader = DataLoader(dataset, shuffle=True)
