import collections
import itertools
import logging
import numpy as np

def get_possible_states(num_price_levels, num_players):
    """
    Function generates a list of all possible states S, where each state is a named tuple of state attributes
    :param num_price_levels: number of discrete prices
    :param num_players: number of players in auction
    :return: list of possible states S
    """
    prices = (np.nan,) + tuple(range(num_price_levels))
    bid_combinations = list(itertools.product(*[prices for p in range(num_players)]))

    State = define_state()
    S = [State(current_winner=np.nan, current_bids=(b)) for b in bid_combinations]
    S = [set_winner(s) for s in S]
    logging.info('Declared {0} possible states, examples: \n{1}'.format(len(S), S[0:3]))
    return S

def get_initial_state(S,initial_state_random):
    if initial_state_random:
        s = np.random.choice(len(S))
        logging.info('Randomly initialize state to {0}'.format({s:S[s]}))
    else:
        s = 0
        logging.info('Set initial state to {0}'.format({s:S[s]}))

    return s


def set_winner(state):
    """
    Function reads in a State named tuple and returns a new State tuple with the field 'current_winner' overwritten
    using the values in the current_bids attribute
    :param state:
    :return:
    """
    state_values = state._asdict()
    state_values['current_winner'] = get_winner(state)
    State = define_state()
    return State(**state_values)

def get_winner(state):
    """
    Function returns a value for the current winner, given the current_bids in the input state
    :param state: input State
    :return: output State with current_winner attribute overwritten
    """
    if all([np.isnan(bid) for bid in state.current_bids]):
        return np.nan
    elif len(state.current_bids) == 1:
        return 0
    elif len(set(state.current_bids)) == 1:
        return np.nan
    else:
        for i, b in enumerate(state.current_bids):
            if b == np.nanmax(state.current_bids):
                return i

def define_state():
    return collections.namedtuple('State', 'current_winner current_bids')


def get_environment_level_file_name(episodes, bid_periods, price_levels, num_players):
    file_name = 'Auction_E{0}_BP{1}_PL{2}_NP{3}'.format(episodes, bid_periods, price_levels, num_players)
    return file_name