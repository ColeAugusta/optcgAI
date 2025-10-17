import pandas as pd
from Card import Card

class Decklist:
    def __init__(self, csv_path, winrate):
        
        self.df = pd.read_csv(csv_path)
        self.prepare_dataframe()
        self.cards = self.create_cards()
        self.winrate = winrate
    
    def prepare_dataframe(self):
        for i in range(5):
            col = str(i)
            if col in self.df.columns:
                self.df[col] = (
                    self.df[col]
                    .astype(str)
                    .str.replace('%', '', regex=False)
                    .astype(float)
                    / 100.0
                )
        self.df['Appearance Rate'] = (
            self.df['Appearance Rate']
            .astype(str)
            .str.replace('%', '', regex=False)
            .astype(float)
            / 100.0
        )

    def create_cards(self):
        cards = []
        for _, row in self.df.iterrows():

            card = Card(
                number = str(row['Card #']),
                card_type = str(row['Type of Card']),
                name = str(row['Card Name']),
                appearance_rate = row['Appearance Rate'],
                decks_appeared = int(row['Appears in # Decks']),
                number_appeared = [row['0'], row['1'], row['2'], row['3'], row['4']]
            )
            cards.append(card)
        return cards