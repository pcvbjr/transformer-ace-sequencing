from shuffles.shuffles import *
from shuffles.decks import *

def test_deck_initialization():
    deck1 = Deck(num_decks=1)
    assert len(deck1.cards) == 52
    card1 = deck1.deal()
    assert isinstance(card1, Card)
    assert len(deck1.cards) == (52 - 1)
    assert len(deck1.discards) == 1

    num_decks = 5
    deck5 = Deck(num_decks=num_decks)
    assert len(deck5.cards) == 52 * num_decks

    deal_times = 13
    for _ in range(deal_times):
        deck5.deal()
    assert len(deck5.cards) == 52 * num_decks - deal_times
    assert len(deck5.discards) == deal_times

def test_generate_deal_sequence():
    shuffle = Shuffle(num_decks=1, shuffle_threshold=0.8)
    deal_seq = shuffle.generate_deal_sequence(12)
    assert len(deal_seq) == 12
    assert deal_seq[11] == 'shuffle'

def test_riffle_pass():
    deck = Deck()
    for _ in range(10000):
        deck.cards = riffle_pass(deck.cards)
        assert len(deck.cards) == 52

def test_strip_pass():
    deck = Deck()
    for _ in range(10000):
        deck.cards = strip_pass(deck.cards)
        assert len(deck.cards) == 52

def test_cut_pass():
    deck = Deck()
    for _ in range(10000):
        deck.cards = cut_pass(deck.cards)
        assert len(deck.cards) == 52