import numpy as np
import environment as env
import logging
import random

class Player(object):
    """
    This is an agent
    """
    def __init__(self, player_id, alpha, gamma, epsilon, epsilon_decay_1, epsilon_decay_2, epsilon_threshold, agent_valuation, S):
        self.player_id = player_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_1 = epsilon_decay_1
        self.epsilon_decay_2 = epsilon_decay_2
        self.epsilon_threshold = epsilon_threshold
        self.agent_valuation = agent_valuation
        self.S = S
        self.state_dict = dict(zip(list(range(len(S))), S))
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

        # once player has bid nan, cannot re-enter auction (after initial period)
        filt3 = np.array(
            [[(np.isnan(x.current_bids[self.player_id])) & (~np.isnan(y.current_bids[self.player_id])) for y in S] for x in S])
        R3D[1:, filt3] = np.nan

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

    def get_available_actions(self,t,s):
        actions = [i for i,v in enumerate(self.R[t,s]) if not np.isnan(v)]
        resulting_states = [self.S[a] for a in actions]
        action_dict = dict(zip(actions,resulting_states))
        logging.debug('Available actions: {0}'.format(action_dict))
        return actions

    def select_action(self,t,s):
        """
        Method selects an action using the current time, state and player epsilon value
        :param t: bidding period
        :param s: current state
        :return: action a
        """
        if np.random.binomial(1, self.epsilon):
            logging.debug('Exploratory policy selected using epsilon = {0}'.format(self.epsilon))
            return self.select_action_exploratory(t,s)
        else:
            logging.debug('Greedy policy selected using epsilon = {0}'.format(self.epsilon))
            return self.select_action_greedy(t,s)

    def select_action_greedy(self,t,s):
        actions = self.get_available_actions(t, s)
        q_values = [(a, q) for a, q in enumerate(self.Q[t, s]) if a in actions]
        qv_max = [(a, q) for a, q in q_values if q == max(x[1] for x in q_values)]
        qv_summary = [({q[0]: self.state_dict[q[0]]}, q[1]) for q in qv_max]
        logging.debug('Highest-valued possible actions are: {0}'.format(qv_summary))
        a = random.choice(qv_max)[0]
        logging.debug('Action {0} selected at random from highest-valued list'.format({a: self.state_dict[a]}))
        return a

    def select_action_exploratory(self,t,s):
        actions = self.get_available_actions(t, s)
        a = np.random.choice(actions)
        logging.debug('Action {0} selected at random.'.format({a:self.state_dict[a]}))
        return a

    def get_reward(self,t,s,a):
        r = self.R[t, s, a]
        logging.info('Reward for player {0} in time period {1} for action {2} from state {3} = {4}'.format(
            self.player_id, t, a, s, r))
        return r

    def update_q(self,t,s,a,is_final_period:bool):
        """
        Agent updates its Q matrix
        """
        # 5) Check from the Q matrix the old Q vlue of the (s,a) pair and the values of the (s',a') that this agent thinks would follow
        Qold = self.Q[t,s,a]

        r = self.get_reward(t,s,a)

        # check available actions and associated q-values for new state: assume greedy policy choice of action for a2
        t2 = t+1 if not is_final_period else t
        s2 = a
        actions2 = [i for i,v in enumerate(self.R[t2,s2]) if not np.isnan(v)]
        q_values2 = [(i,v) for i,v in enumerate(self.Q[t2,s2]) if i in actions2]
        a2 = sorted(q_values2,key=lambda x: x[1], reverse=True)[0][0]
        Qnext = self.Q[t2,s2,a2]

        # 6) Use the Q-learning rule to calcualte the new Q(s,a) value and update the Q matrix accordingly
        Qnew = round(Qold + self.alpha*(r + self.gamma*Qnext - Qold),2)
        logging.info('Q({0},{1}) = {2} using alpha = {3} and gamma = {4}'.format(s,a,Qnew,self.alpha,self.gamma))

        Q = self.Q
        Q[t,s,a] = Qnew
        self.Q = Q
        logging.debug('Updated Q matrix: \n {0}'.format(self.Q))
        return self.Q

    def update_epsilon(self):
        if self.epsilon > self.epsilon_threshold:
            self.epsilon = self.epsilon * self.epsilon_decay_1
        else:
            self.epsilon = self.epsilon * self.epsilon_decay_2
