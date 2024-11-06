import random
import itertools

# Card class
class Card:
    def __init__(self, rank, suit):
        self.rank = rank # 1=Ace, 11=Jack, 12=Queen, 13=King
        self.suit = suit # 0=spades, 1=hearts, 2=diamonds, 3=clubs
        self.name = self.__repr__()
        self.is_ace = True if self.rank == 1 else False

    def __repr__(self):
        rank_map = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}
        suit_map = {0: 'S', 1: 'H', 2: 'D', 3: 'C'}
        rank = rank_map.get(self.rank, str(self.rank))
        suit = suit_map.get(self.suit)
        return rank + suit
        

# Deck class
class Deck:
    def __init__(self, num_decks=1):
        self.num_decks = num_decks
        self.cards = [Card(rank, suit) for rank, suit in itertools.product(range(1,14), range(0,4))] * num_decks
        random.shuffle(self.cards) # "wash" the deck - assume this is completely randomized
        self.total_cards = 52 * num_decks
        self.discards = []


    def deal(self):
        # Cards are dealt from the end of the deck.
        # The discard pile grows by appending cards to its end.
        # This is equivalent to dealing a card from the top of a face-down deck and 
        # discarding onto the top of a face-down discard pile.
        dealt_card = self.cards.pop() 
        self.discards.append(dealt_card) 
        return dealt_card


