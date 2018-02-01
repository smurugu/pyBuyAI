import os
import numpy as np
import logging
import datetime as dt

from market import Market, MarketRandomWalk

import utils

def main():
    # initialise market
    walk_dist = R = np.random.RandomState(3).normal
    walk_dist_params = (0,1)
    mkt = MarketRandomWalk(walk_dist,walk_dist_params)
    logger.info('Generated market with config: {0}'.format(mkt.__dict__))

    for t in range(0,10):
        logger.info('Running process for time period {0}'.format(t))
        mkt_bid_price = mkt.set_bid_price()
        mkt_ask_price = mkt.set_ask_price()
        logger.info('Bid price: {0}, ask price: {1}'.format(mkt_bid_price,mkt_ask_price))

    return


if __name__ == '__main__':
    logger = utils.setup_logging()
    main()

    print('Main end')