import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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


    # need matrix for card features, for corrent tensor mult
    def create_feature_matrix(self):
        features = []
        for card in self.cards:
            # appearance rate, number appeared, deck freq.
            number_appeared = sum(i * card.number_appeared[i] for i in range(5))
            features.append([
                card.appearance_rate,
                number_appeared,
                card.decks_appeared / max(card.decks_appeared for card in self.cards)
            ])
        return torch.tensor(features, dtype=torch.float32)

    def optimize_deck(self, max_copies=4, learning_rate=0.1, iterations=1000, temperature=1.0, anneal_rate=0.995):

        # logits for copies of cards
        logits = nn.Parameter(torch.zeros(self.n_cards, max_copies + 1))

        for i, card in enumerate(self.cards):
            for count in range(max_copies + 1):
                logits.data[i, count] = np.log(card.number_appeared[count] + 1e-8) # idk really
        
        optimizer = optim.Adam([logits], lr=learning_rate)

        best_deck = None
        best_score = float('-inf')

        # loss and backward steps
        for iteration in iterations:
            optimizer.zero_grad()

            # get differentiable card counts with Softmax
            temp = temperature * (anneal_rate ** iteration)
            card_probs = torch.softmax(logits / temp, dim=1)

            # expected # of cards
            counts = torch.arange(max_copies + 1, dtype=torch.float32)
            expected_counts = (card_probs * counts).sum(dim=1)

            # Calculate loss
            # match deck size,
            # match appearance rate distribution,
            # push toward more expected copies i.e. run 4 of a card
            # penalize underusing core cards
            # entropy
            deck_size_loss = (expected_counts.sum() - self.deck_size) ** 2
            appearance_score = (expected_counts * self.features[:, 0]).sum()
            expected_copy_score = (expected_counts * self.features[:, 1]).sum()
            underuse_penalty = torch.relu(self.features[:,0] * max_copies - expected_counts).sum()
            entropy = -(card_probs * torch.log(card_probs + 1e-8)).sum()

            # combined loss
            loss = (deck_size_loss * 10.0 - appearance_score * 5.0 - expected_copy_score * 2.0 + underuse_penalty * 3.0 - entropy * 0.01)
            loss.backward()
            optimizer.step()



if __name__ == "__main__":
    bonney = Decklist("../data/op12BonneyCards.csv", 0.55)
    optimizer = DeckOptimizer(bonney, deck_size=50)
