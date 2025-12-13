# One Piece TCG Deck Optimizer

A PyTorch-based optimization model that generates optimal decklists for the One Piece Trading Card Game using gradient descent and statistical analysis of competitive deck data.

## Features

- **Differentiable Optimization**: Uses Gumbel-Softmax for gradient-based card selection
- **Multi-Objective Loss**: Balances appearance rates, expected copies, deck size constraints, and diversity
- **Data-Driven**: Leverages statistical distributions from competitive decklists
- **Simulated Annealing**: Temperature scheduling for improved convergence

## Usage

```python
from Decklist import Decklist
from DeckOptimizer import DeckOptimizer

# Load deck data
decklist = Decklist('bonney_deck_data.csv', winrate=0.55)

# Create optimizer
optimizer = DeckOptimizer(decklist, deck_size=50)

# Generate optimal deck
optimized_deck = optimizer.optimize_deck(
    max_copies=4,
    learning_rate=0.1,
    iterations=1000,
    temperature=2.0,
    anneal_rate=0.995
)

# Display results
optimizer.print_deck(optimized_deck)
```

## How It Works

1. **Feature Extraction**: Converts card statistics into numerical features
2. **Gradient Descent**: Optimizes card selection using PyTorch
3. **Loss Function**: 
   - Matches target deck size (50 cards)
   - Maximizes appearance rate of selected cards
   - Aligns with expected copy distributions
   - Penalizes underusing high-appearance cards
   - Encourages deck diversity via entropy
4. **Deck Adjustment**: Post-processes to ensure exactly 50 cards

## Parameters

- `max_copies`: Maximum copies per card (default: 4)
- `learning_rate`: Optimization step size (default: 0.1)
- `iterations`: Training iterations (default: 1000)
- `temperature`: Initial Gumbel-Softmax temperature (default: 2.0)
- `anneal_rate`: Temperature decay rate (default: 0.995)
