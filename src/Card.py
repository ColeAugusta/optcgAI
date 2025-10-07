
class Card:
    def __init__(self, number, card_type, name, appearance_rate, decks_appeared, number_appeared):
        self.number = number
        self.card_type = card_type
        self.name = name
        self.appearance_rate = appearance_rate
        self.decks_appeared = decks_appeared
        self.number_appeared = number_appeared
    
    def __str__(self):
        return f"{self.number} {self.card_type} {self.name} {self.appearance_rate} {self.decks_appeared} {self.number_appeared}"