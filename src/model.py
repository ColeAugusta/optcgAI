import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Card import Card
from Decklist import Decklist

# Nueral network optimizer for single leader optimization
# includes input decklist information for multiple leaders later
class DeckOptimizer:
    def __init__(self, decklist, deck_size=50, leader_id=None):

        self.decklist = decklist
        self.deck_size = deck_size
        self.leader_id = leader_id
        self.cards = [card for card in decklist.cards if card.card_type != 'Leader']
        self.n_cards = len(self.cards)
        self.features = self.create_feature_matrix()


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
        for iteration in range(iterations):
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

            # track best deck
            with torch.no_grad():
                hard_counts = card_probs.argmax(dim=1)
                if hard_counts.sum() <= self.deck_size:
                    # card score = appearance + expected - underuse, should change later
                    score = appearance_score + expected_copy_score - underuse_penalty
                    if score > best_score:
                        best_score = score
                        best_deck = hard_counts.clone()
            
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}: Loss = {loss.item():.4f}, " 
                      f"Deck Size = {expected_counts.sum().item():.1f}, "
                      f"Temp = {temp:.4f}")
        
        if best_deck is None:
            best_deck = card_probs.argmax(dim=1)
        
        final_deck = self.adjust_deck_size(best_deck, max_copies)

        return self.create_deck_dict(final_deck)
    
    # need this cause torch tensors, fit to match deck_size
    def adjust_deck_size(self, deck_counts, max_copies):
        current_size = deck_counts.sum().item()
        while current_size < self.deck_size:
            # while curr_size < size, add highest appearance rate, but not max
            scores = []
            for i, card in enumerate(self.cards):
                if deck_counts[i] < max_copies:
                    scores.append((self.features[i, 0].item(), i))
            if not scores:
                break

            scores.sort(reverse=True)
            deck_counts[scores[0][1]] += 1
            current_size += 1
        
        while current_size > self.deck_size:
            # while curr_size > size, remove lowest appearance rate
            scores = []
            for i, card in enumerate(self.cards):
                if deck_counts[i] > 0:
                    scores.append((self.features[i, 0].item(), i))
                if not scores:
                    break
                scores.sort()
                deck_counts[scores[0][1]] -= 1
                current_size -= 1
        
        return deck_counts
    
    # tensor counts to dictionary
    def create_deck_dict(self, deck_counts):
        deck = {}
        for i, card in enumerate(self.cards):
            count = int(deck_counts[i].item())
            if count > 0:
                deck[card.name] = {
                    'count': count,
                    'number': card.number,
                    'type': card.card_type,
                    'appearance_rate': card.appearance_rate
                }
        return deck
    
    def print_deck(self, deck):
        print(f"\n{'='*70}")
        print(f"Optimized Deck (winrate: {self.decklist.winrate:.2%})")
        print(f"\n{'='*70}")

        total_cards = sum(d['count'] for d in deck.values())
        print(f"Total cards: {total_cards}")
        print(f"\n{'Count':<6} {'Card #':<12} {'Type':<12} {'Name':<25} {'Appear %'}")
        print('-' * 70)

        sorted_deck = sorted(deck.items(),
                             key=lambda x: (x[1]['type'], -x[1]['appearance_rate']))
        for name, info in sorted_deck:
            print(f"{info['count']:<6} {info['number']:<12} {info['type']:<12} "
                  f"{name:<25} {info['appearance_rate']:.1%}")            



if __name__ == "__main__":
    bonney = Decklist("../data/op12BonneyCards.csv", 0.55)
    optimizer = DeckOptimizer(bonney, deck_size=50)

    print ("optimizing...")
    optimized = optimizer.optimize_deck(
        max_copies=4,
        learning_rate=0.1,
        iterations=1000,
        temperature=2.0,
        anneal_rate=0.995
    )

    optimizer.print_deck(optimized)
