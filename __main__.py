import os
import numpy as np
import logging
import datetime as dt

from market import Market, MarketRandomWalk
from player import Player

import utils

def main():
    # initialise market
    walk_dist = np.random.RandomState(3).normal
    walk_dist_params = (0,1)
    mkt = MarketRandomWalk(walk_dist,walk_dist_params)
    logger.info('Generated market with config: {0}'.format(mkt.__dict__))

    player1 = Player(starting_cash=1000)

    for t in range(0,10):
        logger.info('Running process for time period {0}'.format(t))
        willing_to_buy = mkt.willing_to_buy()
        willing_to_sell = mkt.willing_to_sell()
        logger.info('Prices available to buy: {0}'.format(willing_to_sell))
        logger.info('Prices available to sell: {0}'.format(willing_to_buy))

        transactions = player1.evaluate_offers(willing_to_buy,willing_to_sell)

    return


if __name__ == '__main__':
    logger = utils.setup_logging()
    main()

    print('Main end')