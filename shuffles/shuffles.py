import math
import random

from .decks import Deck

# Shuffle base class
class Shuffle:
    """
    Each Shuffle class should be initialized with the number of standard 52-card decks to be used and
    when to shuffle a deck (given by a fraction of the deck remaining, below which, the deck will be reshuffled).
    Initialization of the deck is the same, no matter the shuffling method.
    Each Shuffle subclass will have a different shuffling method.
    When a deck is shuffled, the number of times to perform a shuffling procedure should be passed.
    The generate_deal_sequence method is used to generate a sequence of events from dealing the deck, including cards dealt and shuffle events.
    """
    def __init__(self, num_decks, shuffle_threshold):
        self.deck = Deck(num_decks=num_decks)
        self.shuffle_threshold = shuffle_threshold

    def shuffle(self):
        # This will be implemented in subclasses
        pass

    def generate_deal_sequence(self, num_deals):
        deal_sequence = []
        while len(deal_sequence) < num_deals:
            if len(self.deck.cards) / self.deck.total_cards < self.shuffle_threshold:
                self.shuffle()
                deal_sequence.append('shuffle')
            else:
                deal_sequence.append(self.deck.deal().name)
        return deal_sequence
    
    def add_back_discards(self):
        self.deck.cards.extend(self.deck.discards)
        self.deck.discards = []


def riffle_pass(cards):
    """
    Perform a single riffle pass on the given deck.

    :param list cards: A list of cards to be shuffled
    :return: A new list of cards with the riffle pass applied
    :raises ValueError: If the input deck is not a non-empty list
    """
    return gsm_riffle_shuffle(cards)
    # if not isinstance(cards, list) or len(cards) == 0:
    #     raise ValueError("Input deck must be a non-empty list")

    # split_idx = min(len(cards) // 2 + random.randint(0, 3), len(cards))
    # left_half = cards[:split_idx]
    # right_half = cards[split_idx:]

    # if random.random() > 0.5:
    #     # switch right and left halves
    #     left_half, right_half = right_half, left_half

    # new_cards = []
    # l_pointer = 0
    # r_pointer = 0
    # while l_pointer < len(left_half) and r_pointer < len(right_half):
    #     # add from left half
    #     chunk_size = min(random.randint(1, 3), len(left_half) - l_pointer)
    #     new_cards.extend(left_half[l_pointer : l_pointer + chunk_size])
    #     l_pointer += chunk_size

    #     # add from right half
    #     chunk_size = min(random.randint(1, 3), len(right_half) - r_pointer)
    #     new_cards.extend(right_half[r_pointer : r_pointer + chunk_size])
    #     r_pointer += chunk_size

    # if l_pointer < len(left_half):
    #     new_cards.extend(left_half[l_pointer:])

    # if r_pointer < len(right_half):
    #     new_cards.extend(right_half[r_pointer:])

    # return new_cards

def gsm_riffle_shuffle(cards):
    """
    Perform a single GSM-model riffle pass on the given deck.

    :param list cards: A list of cards to be shuffled
    :return: A new list of cards with the riffle pass applied
    :raises ValueError: If the input deck is not a non-empty list
    """
    if not isinstance(cards, list) or len(cards) == 0:
        raise ValueError("Input deck must be a non-empty list")

    # split deck index ~ N(C/2, C/3), where C is len(cards)
    C = len(cards)
    split_idx = math.floor(random.normalvariate(C / 2, C / 3))
    left_hand, right_hand = cards[:split_idx], cards[split_idx:]
    left_hand.reverse(), right_hand.reverse()

    new_deck = []
    while len(left_hand) > 0 or len(right_hand) > 0:
        proportion_in_left_hand = ( len(left_hand) / ( len(left_hand) + len(right_hand) ))
        if random.random() < proportion_in_left_hand:
            new_deck.append( left_hand.pop() )
        else:
            new_deck.append( right_hand.pop() )
    
    return new_deck


def strip_pass(cards):
    """
    Hold the pack in landscape orientation, then pull the top fifth or so (i.e., a fifth of the deck, give or take 3 cards) of the deck off the top, keeping it close to the remainder of the deck, and set it down next to the pack. Then do the same with the next fifth of the deck, placing it on top of what was the top fifth, and so on, until the entire deck has been gone through in this way.

    :param list cards: A list of cards to be shuffled
    :return: A new list of cards with the strip pass applied
    :raises ValueError: If the input deck is not a non-empty list
    """
    if not isinstance(cards, list) or len(cards) == 0:
        raise ValueError("Input deck must be a non-empty list")

    new_cards = []
    chunk_size = len(cards) // 5

    start_idx = 0
    for i in range(5):
        if i < 4:
            end_idx = (i + 1) * chunk_size + random.randint(-3, 3)
            end_idx = min(end_idx, len(cards))
        else:
            end_idx = len(cards)

        chunk = cards[start_idx:end_idx]
        new_cards = chunk + new_cards

        start_idx = end_idx

    return new_cards

def cut_pass(cards):
    """
    Cut the deck randomly into two packets, with a minimum of 4 cards in each packet.
    Then place the top packet at the bottom of the deck and the bottom packet at the top of the deck.

    :param list deck: A list of cards to be shuffled
    :return: A new list of cards with the cut pass applied
    :raises ValueError: If the input deck is not a non-empty list
    """
    if not isinstance(cards, list) or len(cards) == 0:
        raise ValueError("Input deck must be a non-empty list")

    if len(cards) < 8:
        return cards  # cannot cut into two packets with at least 4 cards each

    cut_idx = random.randint(4, len(cards) - 4)
    top_packet = cards[:cut_idx]
    bottom_packet = cards[cut_idx:]

    return bottom_packet + top_packet

class CutOnlyShuffle(Shuffle):
    def shuffle(self):
        """
        The cut only shuffle just cuts once; this is intended as a dummy shuffle.
        """
        self.add_back_discards()

        self.deck.cards = cut_pass(self.deck.cards)


class HomeShuffle(Shuffle):
    def shuffle(self):
        """
        The home shuffle models a casual shuffle, with just riffle passes and a final cut.
        :param int n_riffle: The number of riffle passes to perform
        """
        self.add_back_discards()

        for _ in range(3):
            self.deck.cards = riffle_pass(self.deck.cards)
        self.deck.cards = cut_pass(self.deck.cards)


class RiffleOnlyShuffle(Shuffle):
    def __init__(self, num_riffle_passes=1):
        self.super().__init__()
        self.num_riffle_passes = num_riffle_passes

    def shuffle(self):
        """
        The home shuffle models a casual shuffle, with just riffle passes and a final cut.
        :param int n_riffle: The number of riffle passes to perform
        """
        self.add_back_discards()

        for _ in range(self.num_riffle_passes):
            self.deck.cards = riffle_pass(self.deck.cards)
        

class CasinoHandShuffle(Shuffle):
    def shuffle(self):
        """
        The casino hand shuffle uses the following deal sequence:
        - Three riffle shuffles
        - A strip shuffle
        - One more riffle shuffle
        - The cut
        """
        self.add_back_discards()

        self.deck.cards = riffle_pass(self.deck.cards)
        self.deck.cards = riffle_pass(self.deck.cards)
        self.deck.cards = riffle_pass(self.deck.cards)
        self.deck.cards = strip_pass(self.deck.cards)
        self.deck.cards = riffle_pass(self.deck.cards)
        self.deck.cards = cut_pass(self.deck.cards)

class AutomaticMachineShuffle(Shuffle):
    def shuffle(self):
        """
        The automatic machine shuffle uses a random number generator to pre-determine the order of cards and sets the deck accordingly.
        """
        self.add_back_discards()

        random.shuffle(self.deck.cards)
