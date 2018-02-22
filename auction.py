import collections
import itertools
import logging
import numpy as np


class Auction_env(object):
    """
    This is the auction environment
    """
    def __init__(self, bid_periods, discrete_price_levels, players):
        self.bid_periods = bid_periods
        self.discrete_price_levels = discrete_price_levels
        self.players = players
        self.R = Auction_env(bid_periods, discrete_price_levels, players)

    @staticmethod
    def generate_R(bid_periods, discrete_price_levels, players, player, player_valuation):
        """
        This function can be called by the environment or any agent to create an R matrix using params
        :param bid_periods: Number of periods the auction will run for
        :param discrete_price_levels: Number of prices ([0,n])
        :param players: Number of players bidding in the auction
        :param player: The ID of the player generating the R matrix
        :param player_valuation: The value the player places on the item for sale in the auction
        :return R matrix: Defines rewards given to players for being in a given state
        """
        # note: this generates an array R3D[t,s,a]
        S = Auction_env.declare_states(discrete_price_levels, players)
        R3D = np.array([[[np.nan for y in S] for x in S] for t in np.arange(bid_periods)])
        R3D = np.zeros(np.shape(R3D))

        # can only bid a price higher than previously bid
        filt2 = np.array([[y.current_bids[player] <= max(x.current_bids) for y in S] for x in S])
        R3D[:, filt2] = np.nan
        # logging.info('Set R matrix values: can only bid values > max value already bid OR nan: \n{0}'.format(R3D))

        # cannot change other player states
        filt3 = np.array(
            [[[b1 for i1, b1 in enumerate(x.current_bids) if i1 != player] != [b2 for i2, b2 in
                                                                               enumerate(y.current_bids) if
                                                                               i2 != player]
              for y in S] for x in S])

        R3D[:, filt3] = np.nan
        # logging.info('Set R matrix values: cannot change other player states: \n{0}'.format(R3D))


        # agent payoff set for final period
        filt4 = np.array([[y.current_winner == player for y in S] for x in S])
        values = np.array(
            [[(player_valuation - max(y.current_bids) if y.current_winner == player else 0) for y in S] for x in S])
        R3D[bid_periods - 1, :, :] = np.multiply(R3D[bid_periods - 1, :, :] + 1, values)
        return R3D

    @staticmethod
    def declare_states(discrete_price_levels, players):
        # Declare all possible states: (t,w,(ps)), where t = time/bidding round, w = current winner, (ps) = tuple indicating latest bids
        prices = (np.nan,) + tuple(range(discrete_price_levels))

        bid_combinations = list(itertools.product(*[prices for p in range(players)]))

        State = Auction_env.define_state()
        S = [State(current_winner=np.nan, current_bids=(b)) for b in bid_combinations]
        S = [Auction_env.set_winner(s) for s in S]
        logging.info('Declared {0} possible states, examples: \n{1}'.format(len(S), S[0:3]))
        return S

    @staticmethod
    def set_winner(state):
        """
        Function set_winner: returns the index of the winning bid in the current_bids item of the State tuple
        Returns nan where all bids are nan
        Returns nan where all bids are equal
        Returns the index of the highest bid where there exists one bid greater than all others, ignoring nan values
        """
        state_values = state._asdict()
        if all([np.isnan(bid) for bid in state.current_bids]):
            state_values['current_winner'] = np.nan
        elif len(state.current_bids) == 1:
            state_values['current_winner'] = 0
        elif len(set(state.current_bids)) == 1:
            state_values['current_winner'] = np.nan
        else:
            for i, b in enumerate(state.current_bids):
                if b == np.nanmax(state.current_bids):
                    state_values['current_winner'] = i
        State = Auction_env.define_state()
        return State(**state_values)

    @staticmethod
    def define_state():
        return collections.namedtuple('State', 'current_winner current_bids')
