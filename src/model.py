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
        self.data.append({
            'cards': self.cards_to_tensor(decklist),
            'win_rate': decklist.winrate
        }) 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return {
            'cards': self.data[index]['cards'],
            'win_rate': torch.tensor([self.data[index]['win_rate']], dtype=torch.float32)
        }

    # build tensors form decklist cards
    def cards_to_tensor(self, decklist: Decklist) -> torch.Tensor:
        features = []
        for card in decklist.cards:
            feature_vec = [
                # simple hashing for tensors
                float(hash(card.number) % 10000),
                float(hash(card.card_type) % 1000),
                float(hash(card.name) % 100000),
                float(card.appearance_rate),
                float(card.decks_appeared)
            ] + [float(x) for x in card.number_appeared]

            features.append(feature_vec)
        
        return torch.tensor(features, dtype=torch.float32)

# Nueral network optimizer for single leader optimization
class DeckOptimizer(nn.Module):
    def __init__(self, card_feature_dim=9, hidden_dim=128):
        super().__init__()

        # process each card
        self.card_encoder = nn.Sequential(
            nn.Linear(card_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # get deck-level features
        self.deck_aggregator = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU()
        )

        # get final prediction
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),         # drop 20% of neurons
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, cards):
        # encode card: (batch, num cards, features) -> (batch, num cards, 32)
        card_encodings = self.card_encoder(cards)

        # aggregate cards
        deck_encoding = torch.mean(card_encodings, dim=1) # (batch, 32)
        deck_features = self.deck_aggregator(deck_encoding) # (batch, hidden_dim)

        win_rate = self.predictor(deck_features)
        return win_rate


def TrainModel(leader, deck_data, epochs=50, batch_size=16, lr=0.001):

    print(f"Training model for leader: {leader}")
    dataset = DeckDataset(deck_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DeckOptimizer()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for cards, win_rates in dataloader:
            optimizer.zero_grad()

            # forward pass/predictions
            predictions = model(cards)
            loss = criterion(predictions, win_rates)

            # backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return model

if __name__ == "__main__":
    bonney = Decklist("../data/op12BonneyCards.csv", 0.55)
    bonney_model = TrainModel("Green Bonney", bonney, epochs=50)
