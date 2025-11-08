import torch
import torch.nn as nn
import torch.optim as optim
from Card import Card
from Decklist import Decklist

# Nueral network optimizer for single leader optimization
class DeckOptimizer:
    def __init__(self, decklist, deck_size=50, leader_id=None):

        self.decklist = decklist
        self.deck_size = deck_size
        self.leader_id = leader_id
        self.cards = [card for card in decklist.cards if card.card_type != 'Leader']
        self.n_cards = len(self.cards)


    def create_feature_matrix(self):
        features = []
        for card in self.cards:
            # appearance rate, number appeared, deck freq.
            number_appeared = sum(i * card.number_appeared[i] for i in range(5))
            features.append([
                card.appearance_rate,
                number_appeared,
                card.decks_appeared
            ])

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

def collate_fn(batch):
    return torch.stack(batch)


def TrainModel(leader, decklist, epochs=50, batch_size=16, lr=0.001):

    print(f"Training model for leader: {leader}")
    dataset = DeckDataset(decklist)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # init model
    model = DeckOptimizer()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    target_winrate = torch.tensor([[decklist.winrate]], dtype=torch.float32)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for card_batch in dataloader:
            # card_batch is (batch_size, 9) - need to reshape to (1, batch_size, 9) for model
            cards = card_batch.unsqueeze(0)  # (1, batch_size, 9)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(cards)
            loss = criterion(predictions, target_winrate)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return model

if __name__ == "__main__":
    bonney = Decklist("../data/op12BonneyCards.csv", 0.55)
    optimizer = DeckOptimizer(bonney, deck_size=50)
