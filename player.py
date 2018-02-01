import numpy as np

class Player(object):
    """
    Class defines a market participant who can buy and sell the given commodity

    Attributes:
    """
    def __init__(self,starting_cash=1000):
        self.starting_cash = starting_cash

    def evaluate_offers(self, willing_to_buy:dict, willing_to_sell:dict):
        return {}

