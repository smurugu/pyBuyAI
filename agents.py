import numpy as np
from auction import Auction_env
class Player(object):
    """
    This is an agent
    """
    def __init__(self, player_id, alpha, gamma, epsilon, epsilon_decay_1, epsilon_decay_2, epsilon_threshold, agent_valuation):
        self.player_id = player_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_1 = epsilon_decay_1
        self.epsilon_decay_2 = epsilon_decay_2
        self. epsilon_threshold = epsilon_threshold
        self.agent_valuation = agent_valuation
        self.Q = None
        self.R = None

    def generate_r(self, bid_periods, discrete_price_levels, players, player_id, agent_valuation):
        self.R = Auction_env.generate_R(bid_periods, discrete_price_levels, players, player_id, agent_valuation)

    def generate_q(self):
        self.Q = np.zeros(np.shape(self.R))

    def select_action(self):
        """
        Agent chooses an action
        """
    pass

    def update_q(self):
        """
        Agent updates its Q matrix
        """
    def update_epsilon(self):
        if self.epsilon > self.epsilon_threshold:
            self.epsilon = self.epsilon * self.epsilon_decay_1
        else:
            self.epsilon = self.epsilon * self.epsilon_decay_2