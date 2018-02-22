import numpy as np
import auction as env


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

    def get_r(self, S, bid_periods, agent_valuation=None):
        # allow override of agent_valuation if desired: default to self.value if not
        agent_valuation = self.agent_valuation if agent_valuation is None else agent_valuation

        R3D = np.array([[[np.nan for y in S] for x in S] for t in np.arange(bid_periods)])
        R3D = np.zeros(np.shape(R3D))

        # can only bid a price higher than previously bid
        filt2 = np.array([[y.current_bids[self.player_id] < max(x.current_bids) for y in S] for x in S])
        R3D[:, filt2] = np.nan

        # once player has bid nan, cannot re-enter auction
        filt3 = np.array(
            [[(np.isnan(x.current_bids[self.player_id])) & (~np.isnan(y.current_bids[self.player_id])) for y in S] for x in S])
        R3D[:, filt3] = np.nan

        # cannot change other player states
        filt4 = np.array(
            [[[b1 for i1, b1 in enumerate(x.current_bids) if i1 != self.player_id] != [b2 for i2, b2 in
                                                                               enumerate(y.current_bids) if
                                                                               i2 != self.player_id]
              for y in S] for x in S])
        R3D[:, filt4] = np.nan

        # agent payoff set for final period is (valuation - price) if winner, 0 if not winner
        filt5 = np.triu(np.array([[True for y in S] for x in S]), 1)
        R3D[bid_periods - 1, filt5] = np.nan
        values = np.array(
            [[(agent_valuation - max(y.current_bids) if y.current_winner == self.player_id else 0) for y in S] for x in S])
        R3D[bid_periods - 1, :, :] = np.multiply(R3D[bid_periods - 1, :, :] + 1, values)

        return R3D

    def set_r(self, S, bid_periods, agent_valuation=None):
        # allow override of agent_valuation if desired: default to self.value if not
        agent_valuation = self.agent_valuation if agent_valuation is None else agent_valuation
        self.R = self.get_r(S, bid_periods, agent_valuation)
        return self.R

    def get_q(self):
        return np.zeros(np.shape(self.R))

    def set_q(self):
        self.Q = self.get_q()
        return self.Q

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
