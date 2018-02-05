import numpy as np
import random
import logging
logger = logging.getLogger('log')

class Market(object):
    """
    A market represents the external state of the marketplace experienced by each market player
    It can be considered "the collection of all other market participants"

    Attributes:
        bid_price: The 'market' bid price
        ask_price: The 'market' ask price
        volume: Max volume to buy or sell vs the market per period
        price_upper_bound: Upper bound on the price to trade with the market
        price_lower_bound: Lower bound on the price to trade with the market
    """

    def __init__(self, bid_price=None, ask_price=None, bid_ask_spread=None, volume=1000, price_bounds=[0,100], bid_ask_spread_bounds=[0.1,5]):
        """
        Return a market object
        :param bid_price: bid price to initialise on - None will give random init
        :param ask_price: ask price to initialise on - None will give random init
        :param volume: max volume to buy or sell per period
        :param price_upper_bound:
        :param price_lower_bound:
        """
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.bid_ask_spread = bid_ask_spread
        self.volume = volume
        self.price_bounds = price_bounds
        self.bid_ask_spread_bounds = bid_ask_spread_bounds

    def set_bid_ask_spread(self):
        self.bid_ask_spread = random.uniform(self.bid_ask_spread_bounds[0],self.bid_ask_spread_bounds[1])
        return self.bid_ask_spread

    def set_bid_price(self):
        self.bid_price = random.uniform(self.price_bounds[0],self.price_bounds[1])
        return self.bid_price

    def set_ask_price(self):
        self.ask_price = self.bid_price + self.set_bid_ask_spread()
        return self.ask_price

    def state_bids(self):
        price = self.set_bid_price()
        volume = self.volume
        return {price:volume}

    def state_asks(self):
        price = self.set_ask_price()
        volume = self.volume
        return {price:volume}


class MarketRandomWalk(Market):
    """
    Market with random walk
    """
    def __init__(self,walk_dist,walk_dist_params):
        super(MarketRandomWalk, self).__init__()
        self.walk_dist = walk_dist
        self.walk_dist_params = walk_dist_params

    def set_bid_ask_spread(self):
        """ If blank, initialise using Market price setting method: otherwise random walk from previous value"""
        if self.bid_ask_spread is None:
            self.bid_ask_spread = super(MarketRandomWalk,self).set_bid_ask_spread()
        else:
            spread_change = self.walk_dist(*self.walk_dist_params)
            # take abs() to allow only positive spread
            self.bid_ask_spread = abs(self.bid_ask_spread + spread_change)

        return self.bid_ask_spread

    def set_bid_price(self):
        """ If blank, initialise using Market price setting method: otherwise random walk from previous value"""
        if self.bid_price is None:
            self.bid_price = super(MarketRandomWalk,self).set_bid_price()
        else:
            price_change = self.walk_dist(*self.walk_dist_params)
            self.bid_price = self.bid_price + price_change

        return self.bid_price

    def set_ask_price(self):
        """ Calculated from bid price + bid-ask spread, where both bid price and spread are random walks"""
        if self.bid_price is None:
            self.set_bid_price()
            self.set_ask_price()
        else:
            bid_ask_spread = self.set_bid_ask_spread()
            self.ask_price = self.bid_price + bid_ask_spread

        return self.ask_price

