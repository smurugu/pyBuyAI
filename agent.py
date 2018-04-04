import numpy as np
import pandas as pd
import environment as env
import logging
import random
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pck
import json
import os
import uuid
from cvxopt import matrix, solvers
import environment as env
import math

class Player(object):
    """
    This is an agent
    Default values 0, 0, 0, 0, 0, 0, 0, 0, []
    """
    def __init__(self, player_id=0, alpha=0, gamma=0, epsilon=0, epsilon_decay_1=0, epsilon_decay_2=0, epsilon_threshold=0, agent_valuation=0, S=0, q_convergence_threshold=100, print_directory=r'.',q_update_mode='foe',share_rewards_on_tie=False,file_name_base='game'):
        self.player_id = player_id
        self.print_directory = print_directory
        self.file_name = self.set_serialised_file_name(file_name_base)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_1 = epsilon_decay_1
        self.epsilon_decay_2 = epsilon_decay_2
        self.epsilon_threshold = epsilon_threshold
        self.agent_valuation = agent_valuation
        self.S = S
        self.Q = None
        self.Q2 = None
        self.R = None
        self.path_df = pd.DataFrame(columns=['player_id','episode','bidding_round','bid','prev_state_index','prev_state_label','action_index','alpha','gamma','epsilon','epsilon_decay_1','epsilon_decay_2','epsilon_threshold','reward','periods_since_q_change','q_converged'])
        if type(S) == list:
            self.state_dict = dict(zip(list(range(len(S))), S))
        self.stationaryQ_episodes = 0
        self.q_convergence_threshold = q_convergence_threshold
        self.Q_converged = False
        self.rewards_vector = None
        self.q_update_mode = q_update_mode
        self.share_rewards_on_tie = share_rewards_on_tie

    def calc_final_reward(self,won_auction,price_paid,agent_valuation,is_tie):
        """
        Calculate rewards from auction.
        If no win, return nothing.
        If tied win, refer to settings on whether to share rewards
        If single winner, return winnings
        :param won_auction: bool
        :param price_paid: bid amount won with
        :param agent_valuation:
        :param is_tie: bool
        :return: reward
        """
        if not won_auction:
            return 0

        if not is_tie:
            return (agent_valuation - price_paid)
        else:
            if self.share_rewards_on_tie:
                return (agent_valuation - price_paid) / 2
            else:
                return 0

    def get_possible_bids(self):
        action_tuples = [s.current_bids for s in self.S]
        actions = sorted(list(set([cb[self.player_id] for cb in action_tuples])))
        return actions

    def get_q2(self, S, bid_periods, agent_valuation):
        #set up a numpy array, dimensions [t,s,a,a]
        actions = self.get_possible_bids()
        q2 = np.zeros((bid_periods,len(S),len(actions),len(actions)))
        return q2

    def set_q2(self, S, bid_periods, agent_valuation=None):
        self.Q2 = self.get_q2(S, bid_periods, agent_valuation)
        return self.Q

    def get_r(self, S, bid_periods, agent_valuation=None):
        # allow override of agent_valuation if desired: default to self.value if not
        agent_valuation = self.agent_valuation if agent_valuation is None else agent_valuation

        R3D = np.array([[[np.nan for y in S] for x in S] for t in np.arange(bid_periods)])
        R3D = np.zeros(np.shape(R3D))

        # can only bid a price higher than previously bid
        filt2 = np.array([[y.current_bids[self.player_id] < max(x.current_bids) for y in S] for x in S])
        R3D[:, filt2] = np.nan

        # NEW: cannot bid the same as another player
        #filt2 = np.array([[y.current_bids[self.player_id] in x.current_bids for y in S] for x in S])
        #R3D[:, filt2] = np.nan

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
        R3D[bid_periods - 1] = np.zeros(np.shape(R3D[bid_periods - 1]))
        filt5 = np.triu(np.array([[True for y in S] for x in S]), 1)
        filt6 = np.tril(np.array([[True for y in S] for x in S]), -1)
        R3D[bid_periods - 1, filt5] = np.nan
        R3D[bid_periods - 1, filt6] = np.nan
        values = np.array(
            [[(agent_valuation - np.nanmax(y.current_bids)  if y.current_winner == self.player_id else 0) for y in S] for x in S])
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

    def get_available_actions_old(self,t,s):
        actions = [i for i,v in enumerate(self.R[t,s]) if not np.isnan(v)]
        resulting_states = [self.S[a] for a in actions]
        action_dict = dict(zip(actions,resulting_states))
        logging.info('Player {0}: Available actions: {1}'.format(self.player_id,action_dict))
        return actions

    def get_available_actions_old(self,t,s):
        actions = [i for i,v in enumerate(self.R[t,s]) if not np.isnan(v)]
        resulting_states = [self.S[a] for a in actions]
        action_dict = dict(zip(actions,resulting_states))
        logging.info('Player {0}: Available actions: {1}'.format(self.player_id,action_dict))
        return actions

    def get_available_actions(self,t,s):
        all_possible_bids = self.get_possible_bids()
        current_bid = self.S[s].current_bids[self.player_id]
        other_bid = self.S[s].current_bids[1-self.player_id]

        if t == 0:
            possible_bids = all_possible_bids
        else:
            #possible_bids = [np.nan] + [b for b in all_possible_bids if b >= current_bid]
            possible_bids = [np.nan] + [b for b in all_possible_bids if b >= max(self.S[s].current_bids)]

        current_state = self.S[s]
        if self.player_id==0:
            result_bids = [(b,other_bid) for b in possible_bids]
        else:
            result_bids = [(other_bid,b) for b in possible_bids]

        possible_result_states = [self.S[s]._replace(current_bids=b) for b in result_bids ]
        possible_result_states = [st._replace(current_winner=env.get_winner_for_state(st)) for st in possible_result_states]
        actions = [self.S.index(st) for st in possible_result_states]
        action_dict = dict(zip(actions,possible_result_states))
        logging.info('Player {0}: Available actions: {1}'.format(self.player_id, action_dict))
        return actions

    def select_action(self,t,s):
        """
        Method selects an action using the current time, state and player epsilon value
        :param t: bidding period
        :param s: current state
        :return: action a
        """
        if np.random.binomial(1, self.epsilon):
            logging.info('Player {0}: Exploratory policy selected using epsilon = {1}'.format(self.player_id,self.epsilon))
            return self.select_action_exploratory(t,s)
        else:
            logging.info('Player {0}: Greedy policy selected using epsilon = {1}'.format(self.player_id,self.epsilon))
            return self.select_action_greedy(t,s)

    def select_action_greedy_old(self,t,s):
        actions = self.get_available_actions(t, s)
        q_values = [(a, q) for a, q in enumerate(self.Q[t, s]) if a in actions]
        qv_max = [(a, q) for a, q in q_values if q == max(x[1] for x in q_values)]
        qv_summary = [({q[0]: self.state_dict[q[0]]}, q[1]) for q in qv_max]
        logging.info('Player {0}: Highest-valued possible actions are: {1}'.format(self.player_id, qv_summary))
        a = random.choice(qv_max)[0]
        logging.info('Player {0}: Action {1} selected at random from highest-valued list'.format(self.player_id, {a: self.state_dict[a]}))
        return a

    def select_action_greedy(self,t,s):
        all_bids = self.get_possible_bids()
        actions = self.get_available_actions(t, s)
        qpayoff_matrix = self.Q2[t,s]

        other_player_current_bid=self.S[s].current_bids[1-self.player_id]
        other_player_current_bid_index=all_bids.index(other_player_current_bid)

        #assume other player will stick with current bid
        q_payoff = qpayoff_matrix[:,other_player_current_bid_index]
        max_q = np.nanmax(q_payoff)
        max_locs = [x[0] for x in np.argwhere(q_payoff == max_q).tolist()]
        qv_max = [(a, max_q) for a in max_locs]

        #assume all p2 actions equally likely and pick the row with the highest value
        #q_values = np.sum(qpayoff_matrix, axis=1) / np.shape(qpayoff_matrix)[1]
        #max_q = np.nanmax(qpayoff_matrix)
        #max_locs = [x[0] for x in np.argwhere(qpayoff_matrix==max_q).tolist()]
        #qv_max = [(a,max_q) for a in max_locs]

        current_state = self.S[s]
        current_bids = current_state.current_bids

        highest_valued_new_bids = [all_bids[q[0]] for q in qv_max]
        new_bid_qvalues = [q[1] for q in qv_max]
        corresponding_bid_statuses = [tuple(h if i ==self.player_id else b for i,b in enumerate(current_bids)) for h in highest_valued_new_bids]
        corresponding_new_states = [current_state._replace(current_bids=cb) for cb in corresponding_bid_statuses]
        corresponding_new_states = [s._replace(current_winner=env.get_winner_for_state(s)) for s in corresponding_new_states]

        corresponding_new_state_indices = [self.S.index(ns) for ns in corresponding_new_states]
        qv_summary = dict(zip(corresponding_new_states,new_bid_qvalues))

        logging.info('Player {0}: Highest-valued possible actions are: {1}'.format(self.player_id, qv_summary))
        a = random.choice(corresponding_new_state_indices)
        logging.info('Player {0}: Action {1} selected at random from highest-valued list'.format(self.player_id, {a: self.state_dict[a]}))
        return a

    def select_action_maximin(self,t,s):
        actions = self.get_available_actions(t, s)
        qpayoff_matrix = self.Q2[t,s]

        current_state = self.S[s]
        current_bids = current_state.current_bids
        all_bids=self.get_possible_bids()
        highest_valued_new_bids = [all_bids[b] for b in [solve_maximin(qpayoff_matrix)[0]]]
        new_bid_qvalues = [solve_maximin(qpayoff_matrix)[1]]
        corresponding_bid_statuses = [tuple(h if i ==self.player_id else b for i,b in enumerate(current_bids)) for h in highest_valued_new_bids]
        corresponding_new_states = [current_state._replace(current_bids=cb) for cb in corresponding_bid_statuses]
        corresponding_new_states = [s._replace(current_winner=env.get_winner_for_state(s)) for s in corresponding_new_states]

        corresponding_new_state_indices = [self.S.index(ns) for ns in corresponding_new_states]
        qv_summary = dict(zip(corresponding_new_states,new_bid_qvalues))

        logging.info('Player {0}: Highest-valued possible actions are: {1}'.format(self.player_id, qv_summary))
        a = random.choice(corresponding_new_state_indices)
        logging.info('Player {0}: Action {1} selected at random from highest-valued list'.format(self.player_id, {a: self.state_dict[a]}))
        return a

    def select_action_exploratory(self,t,s):
        actions = self.get_available_actions(t, s)
        a = np.random.choice(actions)
        logging.info('Action {0} selected at random.'.format({a:self.state_dict[a]}))
        return a

    def get_reward(self,t,s,a):
        r = self.R[t, s, a]
        return r

    def get_qpayoff_matrix(self, t, s):
        """
        Function returns an action*action grid of q values for the given player
        :param t:
        :param s:
        :return:
        """
        action_tuples = [s.current_bids for s in self.S]
        #reorder action_tuples to be from players perspective
        action_tuples = [(x[self.player_id],)+tuple(y for i,y in enumerate(x) if i !=self.player_id) for x in action_tuples]
        action_qvalues = dict(zip(action_tuples,self.Q[t, s]))
        action_rpossible = dict(zip(action_tuples, np.divide(self.R[t,s]+0.001,self.R[t,s]+0.001)))
        players = len(action_tuples[0])
        actions_p0 = list(set([cb[0] for cb in action_tuples]))

        qpayoff_matrix = np.zeros((len(actions_p0),)*players)
        for bid_combo in action_qvalues:
            bid_combo_ind = tuple(actions_p0.index(b) for b in bid_combo)
            qpayoff_matrix[bid_combo_ind] = action_qvalues[bid_combo] * action_rpossible[bid_combo]

        if np.sum(self.Q[t,s] !=0) > 1:
            print('Player is {}, time is {}'.format(self.player_id,t))
            print('Current state is: {}'.format(self.S[s]))
            print('Most valuable next state with value {} is: {}'.format(np.max(self.Q[t,s]),self.S[np.argmax(self.Q[t,s])]))
            print(qpayoff_matrix)




            print('lala more rows')
        return qpayoff_matrix

    def get_payoff_matrix_notused(self,t):
        """
        Function returns an action*action grid of q values for the given player
        :param t:
        :param s:
        :return:
        """
        action_tuples = [s.current_bids for s in self.S]
        #reorder action_tuples to be from players perspective
        action_tuples = [(x[self.player_id],)+tuple(y for i,y in enumerate(x) if i !=self.player_id) for x in action_tuples]
        payoff_qvalues = list(np.diag(self.Q[t]))
        action_qvalues = dict(zip(action_tuples,payoff_qvalues))
        players = len(action_tuples[0])
        actions_p0 = list(set([cb[0] for cb in action_tuples]))

        qvalue_array = np.zeros((len(actions_p0),)*players)
        for bid_combo in action_tuples:
            bid_combo_ind = tuple(actions_p0.index(b) for b in bid_combo)
            qvalue_array[bid_combo_ind] = action_qvalues[bid_combo]

        return qvalue_array

    def update_q(self,t,s,actions_taken,is_final_period:bool):
        if self.q_update_mode == 'qlearn':
            return self.update_q_qlearning(t,s,actions_taken,is_final_period)
        elif self.q_update_mode == 'friend':
            return self.update_q_friends(t,s,actions_taken,is_final_period)
        elif self.q_update_mode == 'foe':
            return self.update_q_foe(t,s,actions_taken,is_final_period)
        else:
            logging.error("q_update_mode parameter not accepted: '{}'. Accepted values: " \
                          "'qlearn','friend','foe'")

    def update_q_qlearning(self,t,s,actions_taken,is_final_period:bool):

        possible_bids = self.get_possible_bids()
        # 5) Check from the Q matrix the old Q vlue of the (s,a) pair and the values of the (s',a') that this agent thinks would follow
        a1 = actions_taken[self.player_id]['action']
        price_paid1 = self.S[a1].current_bids[self.player_id]
        a1 = possible_bids.index(price_paid1)
        a2 = actions_taken[1-self.player_id]['action']
        price_paid2 = self.S[a2].current_bids[1-self.player_id]
        a2 = possible_bids.index(price_paid2)

        #TEMP: try setting s to what it was at the time action was taken
        s = actions_taken[self.player_id]['state']

        Qold = self.Q2[t,s,a1,a2]
        # if bidding has ended, current player has highest bid (setting nans to -1) and current player bid is not nan
        won_auction = (~np.isnan(price_paid1)) & ((price_paid1 >= price_paid2) | np.isnan(price_paid2))
        is_tie=env.get_winner((price_paid1,price_paid2))
        r = self.calc_final_reward(won_auction,price_paid1,self.agent_valuation,is_tie)

        logging.info('Reward for player {0} in time period {1} for action {2} from state {3} = {4}'.format(
            self.player_id, t, a1, s, r))

        Q_curr = self.Q2[t,s,:,:]
        Q_curr = np.reshape(Q_curr,(len(possible_bids),len(possible_bids)))
        logging.debug(
            'Player {}: Q matrix for period {} from state {}: \n {}'.format(self.player_id,t,self.S[s],Q_curr))

        t2 = t + 1 if not is_final_period else t
        final_action = [actions_taken[x]['action'] for x in actions_taken if actions_taken[x]['order'] == 1][0]
        s2 = final_action
        #s2 = actions_taken[-1:][0]
        Q_next = self.Q2[t2,s2,:,:]


        Q_next = np.reshape(Q_next,(len(possible_bids),len(possible_bids)))
        V_next = np.max(self.Q2[t2,s2,:,a2])
        #logging.debug('Player {}: Value {} extracted from Q matrix for next period {} from state {}: \n {}'.format(self.player_id,V_next,t2,self.S[s2],Q_next))

        # 6) Use the Q-learning rule to calcualte the new Q(s,a) value and update the Q matrix accordingly
        Qnew = round(Qold + self.alpha*(r + self.gamma*V_next - Qold),2)
        logging.info('Player {0}: using alpha = {6} and gamma = {7}, Q({1},{2},{3},{4}) = {5}'.format(
            self.player_id,t,s,a1,a1,Qnew,self.alpha,self.gamma))

        if Qnew==Qold:
            self.add_to_stationaryQ_episodes()
        else:
            self.reset_stationaryQ_episodes()

        if self.stationaryQ_episodes > self.q_convergence_threshold:
            #set convergence status to true if q matrix not changed for x periods
            # do not reset to False again, even if q matrix is later updated
            self.set_Q_converged(True)

        Q = self.Q2
        Q[t,s,a1,a2] = Qnew
        #Q[t, s, a2, a1] = Qnew #try flipping dimensions
        self.Q2 = Q
        #logging.debug('Updated Q matrix: \n {0}'.format(self.Q2))
        return self.Q2


    def update_q_qlearning_old(self,t,s,a,is_final_period:bool):
        """
        Agent updates its Q matrix
        """
        # 5) Check from the Q matrix the old Q vlue of the (s,a) pair and the values of the (s',a') that this agent thinks would follow
        Qold = self.Q[t,s,a]

        r = self.get_reward(t,s,a)
        logging.info('Player {0}: Reward in time period {1} for action {2} from state {3} = {4}'.format(
            self.player_id, t, a, s, r))

        # check available actions and associated q-values for new state: assume greedy policy choice of action for a2
        t2 = t+1 if not is_final_period else t
        s2 = a
        logging.info('Player {0}: Updating Q value (evaluating selection of greedy action for T+1)'.format(self.player_id))
        a2 = self.select_action_greedy(t2,s2)
        Qnext = self.Q[t2,s2,a2]
        V = Qnext
        # 6) Use the Q-learning rule to calcualte the new Q(s,a) value and update the Q matrix accordingly
        Qnew = round(Qold + self.alpha*(r + self.gamma*V - Qold),2)
        logging.info('Player {0}: Q({1},{2}) = {3} using alpha = {4} and gamma = {5}'.format(self.player_id,s,a,Qnew,self.alpha,self.gamma))

        if Qnew==Qold:
            self.add_to_stationaryQ_episodes()
        else:
            self.reset_stationaryQ_episodes()

        if self.stationaryQ_episodes > self.q_convergence_threshold:
            #set convergence status to true if q matrix not changed for x periods
            # do not reset to False again, even if q matrix is later updated
            self.set_Q_converged(True)

        Q = self.Q
        Q[t,s,a] = Qnew
        self.Q = Q
        #logging.debug('Updated Q matrix: \n {0}'.format(self.Q))
        return self.Q

    def update_q_friends(self,t,s,actions_taken,is_final_period:bool):

        possible_bids = self.get_possible_bids()
        # 5) Check from the Q matrix the old Q vlue of the (s,a) pair and the values of the (s',a') that this agent thinks would follow
        a1 = actions_taken[self.player_id]['action']
        price_paid1 = self.S[a1].current_bids[self.player_id]
        a1 = possible_bids.index(price_paid1)
        a2 = actions_taken[1-self.player_id]['action']
        price_paid2 = self.S[a2].current_bids[1-self.player_id]
        a2 = possible_bids.index(price_paid2)

        #TEMP: try setting s to what it was at the time action was taken
        s = actions_taken[self.player_id]['state']

        Qold = self.Q2[t,s,a1,a2]

        # if bidding has ended, current player has highest bid (setting nans to -1) and current player bid is not nan
        won_auction = (~np.isnan(price_paid1)) & ((price_paid1 >= price_paid2) | np.isnan(price_paid2))
        is_tie=env.get_winner((price_paid1,price_paid2))
        r = self.calc_final_reward(won_auction,price_paid1,self.agent_valuation,is_tie)

        logging.info('Reward for player {0} in time period {1} for action {2} from state {3} = {4}'.format(
            self.player_id, t, a1, s, r))

        Q_curr = self.Q2[t,s,:,:]
        Q_curr = np.reshape(Q_curr,(len(possible_bids),len(possible_bids)))
        logging.debug(
            'Player {}: Q matrix for period {} from state {}: \n {}'.format(self.player_id,t,self.S[s],Q_curr))

        t2 = t + 1 if not is_final_period else t
        final_action = [actions_taken[x]['action'] for x in actions_taken if actions_taken[x]['order'] == 1][0]
        s2 = final_action
        #s2 = actions_taken[-1:][0]
        Q_next = self.Q2[t2,s2,:,:]

        Q_next = np.reshape(Q_next,(len(possible_bids),len(possible_bids)))
        if is_final_period:
            V_next=np.max(Q_curr)
        else:
            V_next = np.max(Q_next)
        #logging.debug('Player {}: Value {} extracted from Q matrix for next period {} from state {}: \n {}'.format(self.player_id,V_next,t2,self.S[s2],Q_next))

        # 6) Use the Q-learning rule to calcualte the new Q(s,a) value and update the Q matrix accordingly
        Qnew = round(Qold + self.alpha*(r + self.gamma*V_next - Qold),2)
        logging.info('Player {0}: using alpha = {6} and gamma = {7}, Q({1},{2},{3},{4}) = {5}'.format(
            self.player_id,t,s,a1,a1,Qnew,self.alpha,self.gamma))

        if Qnew==Qold:
            self.add_to_stationaryQ_episodes()
        else:
            self.reset_stationaryQ_episodes()

        if self.stationaryQ_episodes > self.q_convergence_threshold:
            #set convergence status to true if q matrix not changed for x periods
            # do not reset to False again, even if q matrix is later updated
            self.set_Q_converged(True)

        Q = self.Q2
        Q[t,s,a1,a2] = Qnew
        #Q[t, s, a2, a1] = Qnew #try flipping dimensions
        self.Q2 = Q
        #logging.debug('Updated Q matrix: \n {0}'.format(self.Q2))
        return self.Q2



    def update_q_friends_old(self,t,s,a,is_final_period:bool):
        """
        Agent updates its Q matrix according to a friends rule,
        ie the value of a state is the max Q value for the current state over all possible actions from all players

        """
        # 5) Check from the Q matrix the old Q vlue of the (s,a) pair and the values of the (s',a') that this agent thinks would follow
        Qold = self.Q[t,s,a]

        r = self.get_reward(t,s,a)
        logging.info('Player {0}: Reward in time period {1} for action {2} from state {3} = {4}'.format(
            self.player_id, t, a, s, r))

        # check available actions and associated q-values for new state: assume greedy policy choice of action for a2
        t2 = t+1 if not is_final_period else t
        s2 = a
        logging.info('Updating Q value (evaluating selection of greedy action for T+1)')
        V = max(self.Q[t2,s2]) #Vi(s) = max_a( Qi(s,a) ), where a represents the vector of actions by all players
        #a2 = self.select_action_greedy(t2,s2)
        #Qnext = self.Q[t2,s2,a2]

        # 6) Use the Q-learning rule to calcualte the new Q(s,a) value and update the Q matrix accordingly
        Qnew = round(Qold + self.alpha*(r + self.gamma*V - Qold),2)
        logging.info('Q({0},{1}) = {2} using alpha = {3} and gamma = {4}'.format(s,a,Qnew,self.alpha,self.gamma))

        if Qnew==Qold:
            self.add_to_stationaryQ_episodes()
        else:
            self.reset_stationaryQ_episodes()

        if self.stationaryQ_episodes > self.q_convergence_threshold:
            #set convergence status to true if q matrix not changed for x periods
            # do not reset to False again, even if q matrix is later updated
            self.set_Q_converged(True)

        Q = self.Q
        Q[t,s,a] = Qnew
        self.Q = Q
        #logging.debug('Updated Q matrix: \n {0}'.format(self.Q))
        return self.Q

    def update_q_foe_old(self,t,s,a,is_final_period:bool):
        """
        Agent updates its Q matrix according to a friends rule,
        ie the value of a state is the max Q value for the current state over all possible actions from all players

        """
        # 5) Check from the Q matrix the old Q vlue of the (s,a) pair and the values of the (s',a') that this agent thinks would follow
        Qold = self.Q[t,s,a]

        r = self.get_reward(t,s,a)
        logging.info('Reward for player {0} in time period {1} for action {2} from state {3} = {4}'.format(
            self.player_id, t, a, s, r))

        # check available actions and associated q-values for new state: assume greedy policy choice of action for a2
        t2 = t+1 if not is_final_period else t
        s2 = a
        logging.info('Updating Q value (evaluating selection of greedy action for T+1)')

        current_Q = self.get_qpayoff_matrix(t2, s2)
        if np.sum((current_Q != 0) & (~np.isnan(current_Q))) > 1:
            print('Argh!')
        prime_objective = solve_maximin(current_Q)
        print(prime_objective)
        V = prime_objective #Vi(s) = max_a( Qi(s,a) ), where a represents the vector of actions by all players
        #a2 = self.select_action_greedy(t2,s2)
        #Qnext = self.Q[t2,s2,a2]

        # 6) Use the Q-learning rule to calcualte the new Q(s,a) value and update the Q matrix accordingly
        Qnew = round(Qold + self.alpha*(r + self.gamma*V - Qold),2)
        logging.info('Q({0},{1}) = {2} using alpha = {3} and gamma = {4}'.format(s,a,Qnew,self.alpha,self.gamma))

        if Qnew==Qold:
            self.add_to_stationaryQ_episodes()
        else:
            self.reset_stationaryQ_episodes()

        if self.stationaryQ_episodes > self.q_convergence_threshold:
            #set convergence status to true if q matrix not changed for x periods
            # do not reset to False again, even if q matrix is later updated
            self.set_Q_converged(True)

        Q = self.Q
        Q[t,s,a] = Qnew
        self.Q = Q
        #logging.debug('Updated Q matrix: \n {0}'.format(self.Q))
        return self.Q


    def update_q_foe(self,t,s,actions_taken,is_final_period:bool):
        """
        Agent updates its Q matrix according to a foe rule,
        ie the value of a state is the max Q value for the current state over all possible actions from all players
        """
        possible_bids = self.get_possible_bids()
        # 5) Check from the Q matrix the old Q vlue of the (s,a) pair and the values of the (s',a') that this agent thinks would follow
        a1 = actions_taken[self.player_id]['action']
        price_paid1 = self.S[a1].current_bids[self.player_id]
        a1 = possible_bids.index(price_paid1)
        a2 = actions_taken[1-self.player_id]['action']
        price_paid2 = self.S[a2].current_bids[1-self.player_id]
        a2 = possible_bids.index(price_paid2)

        #TEMP: try setting s to what it was at the time action was taken
        s = actions_taken[self.player_id]['state']

        Qold = self.Q2[t,s,a1,a2]
        # if bidding has ended, current player has highest bid (setting nans to -1) and current player bid is not nan
        won_auction = (~np.isnan(price_paid1)) & ((price_paid1 >= price_paid2) | np.isnan(price_paid2))
        is_tie=env.get_winner((price_paid1,price_paid2))
        r = self.calc_final_reward(won_auction,price_paid1,self.agent_valuation,is_tie)

        logging.info('Reward for player {0} in time period {1} for action {2} from state {3} = {4}'.format(
            self.player_id, t, a1, s, r))

        Q_curr = self.Q2[t,s,:,:]
        Q_curr = np.reshape(Q_curr,(len(possible_bids),len(possible_bids)))
        logging.debug(
            'Player {}: Q matrix for period {} from state {}: \n {}'.format(self.player_id,t,self.S[s],Q_curr))

        t2 = t + 1 if not is_final_period else t
        final_action = [actions_taken[x]['action'] for x in actions_taken if actions_taken[x]['order'] == 1][0]
        s2 = final_action
        #s2 = actions_taken[-1:][0]
        Q_next = self.Q2[t2,s2,:,:]


        Q_next = np.reshape(Q_next,(len(possible_bids),len(possible_bids)))
        V_next = solve_maximin(Q_next)[1]
        #logging.debug('Player {}: Value {} extracted from Q matrix for next period {} from state {}: \n {}'.format(self.player_id,V_next,t2,self.S[s2],Q_next))

        # 6) Use the Q-learning rule to calcualte the new Q(s,a) value and update the Q matrix accordingly
        Qnew = round(Qold + self.alpha*(r + self.gamma*V_next - Qold),2)
        logging.info('Player {0}: using alpha = {6} and gamma = {7}, Q({1},{2},{3},{4}) = {5}'.format(
            self.player_id,t,s,a1,a1,Qnew,self.alpha,self.gamma))

        if Qnew==Qold:
            self.add_to_stationaryQ_episodes()
        else:
            self.reset_stationaryQ_episodes()

        if self.stationaryQ_episodes > self.q_convergence_threshold:
            #set convergence status to true if q matrix not changed for x periods
            # do not reset to False again, even if q matrix is later updated
            self.set_Q_converged(True)

        Q = self.Q2
        Q[t,s,a1,a2] = Qnew
        #Q[t, s, a2, a1] = Qnew #try flipping dimensions
        self.Q2 = Q
        #logging.debug('Updated Q matrix: \n {0}'.format(self.Q2))
        return self.Q2

    def add_to_stationaryQ_episodes(self):
        self.stationaryQ_episodes += 1
        return self.stationaryQ_episodes

    def reset_stationaryQ_episodes(self):
        self.stationaryQ_episodes = 0
        return self.stationaryQ_episodes

    def set_Q_converged(self, converged:bool):
        self.Q_converged = converged
        return self.Q_converged

    def update_epsilon(self,rounding_amt=7):
        if self.epsilon > self.epsilon_threshold:
            epsilon = self.epsilon * self.epsilon_decay_1
        else:
            epsilon = self.epsilon * self.epsilon_decay_2

        self.epsilon = round(epsilon,rounding_amt)

    def get_path_log_entry(self, episode, bidding_round, prev_state_index, action_index):
        #'episode','bidding_round','prev_state_index','prev_state_label','action_index','bid','alpha','gamma','epsilon','reward'
        row_df = pd.DataFrame(index=[0],columns=self.path_df.columns)
        for col in ['episode','bidding_round','prev_state_index','action_index']:
            row_df[col] = locals()[col]

        for col in ['alpha','gamma','epsilon','epsilon_decay_1','epsilon_decay_2','epsilon_threshold','agent_valuation']:
            row_df[col] = self.__getattribute__(col)

        row_df['prev_state_label'] = str(self.S[prev_state_index])
        row_df['bid'] = self.S[action_index].current_bids[self.player_id]
        row_df['reward'] = self.get_reward(bidding_round, prev_state_index, action_index)
        row_df['periods_since_q_change'] = self.stationaryQ_episodes
        row_df['q_converged'] = self.Q_converged
        row_df['player_id'] = self.player_id

        return row_df[self.path_df.columns]

    def write_path_log_entry(self, csv_path=None, log_args=()):

        if csv_path is None:
            csv_path = self.get_serialised_file_name() + '.hdf'

        if os.path.isfile(csv_path):
            # write single row only
            f = open(csv_path, "a+")
            entry = '\n' + '#'.join([str(x) for x in self.get_path_log_entry(*log_args).values[0]])
            f.write(entry)
            f.close()
        else:
            f = open(csv_path, "w+")
            entry = '#'.join(self.path_df.columns)
            f.write(entry)
            f.close()
            self.write_path_log_entry(csv_path=csv_path, log_args=log_args)

        return

    def update_path_log(self, episode, bidding_round, prev_state, action):
        self.path_df = self.path_df.append(self.get_path_log_entry(episode, bidding_round, prev_state, action))
        return self.path_df

    def get_path_log_from_hdf(self,hdf_file):

        return pd.read_csv(hdf_file,sep='#')

    def get_serialised_file_name_old(self):
        T,S,A = np.shape(self.R)
        file_name = 'player{0}_T{1}_S{2}_a{3}_g{4}_eth{5}_ed1-{6}_ed2-{7}'.format(
            self.player_id,
            T,
            S,
            self.alpha,
            self.gamma,
            self.epsilon_threshold,
            self.epsilon_decay_1,
            self.epsilon_decay_2
        ).replace('.','')
        env.check_and_create_directory(self.print_directory)
        file_name = os.path.join(self.print_directory,file_name)
        return file_name

    def set_serialised_file_name_old(self):
        file_name = str(uuid.uuid4())
        env.check_and_create_directory(self.print_directory)
        self.file_name = os.path.join(self.print_directory, file_name)
        return self.file_name

    def set_serialised_file_name(self,file_name_base):
        file_name = 'player_'+str(self.player_id)+'_'+str(file_name_base)
        env.check_and_create_directory(self.print_directory)
        self.file_name = os.path.join(self.print_directory, file_name)
        return self.file_name

    def get_serialised_file_name(self):
        return self.file_name

    def serialise_agent(self):
        """
        Function saves down the metadata, matrices and parameters of a player
        States must be serialised as strings
        :return:
        """
        file_name = self.get_serialised_file_name()

        S2 = []
        for i,v in enumerate(self.S):
            S2 = S2 + [str(v)]

        state_dict2 = {}
        for i,k in enumerate(self.state_dict):
            state_dict2[k] = str(self.state_dict[k])

        d = self.__dict__.copy()
        d['S'] = S2
        d['state_dict'] = state_dict2

        try:
            np.save(file_name,d)
            logging.info('Serialised Player {0} to file: {1} successfully'.format(self.player_id,file_name))
            return file_name
        except Exception as ex:
            logging.error('Failed to serialise player {0} to file: {1}'.format(self.player_id,ex))
            return False

    def load_serialised_agent(self,file_name):

        agent_data = np.load(file_name)[()]

        for attr in agent_data:
            self.__setattr__(attr,agent_data[attr])

        return self

def solve_maximin(q):
    expected_enemy_action = np.argmin(np.sum(q,axis=0))
    retaliatory_action = np.argmin(q[:,expected_enemy_action])

    value = q[retaliatory_action,expected_enemy_action]
    return (retaliatory_action,value)

def solve_maximin_rand(q):
    expected_enemy_action = np.argmin(np.sum(q,axis=0))
    max_return_after_action = np.nanmax(q[:,expected_enemy_action])
    max_locs = [x[0] for x in np.argwhere(q[:,expected_enemy_action] == max_return_after_action).tolist()]
    retaliatory_action = np.random.choice(max_locs)

    value = q[retaliatory_action,expected_enemy_action]
    return (retaliatory_action,value)

def solve_maximin_old(q):

    glpksolver = 'glpk'
    solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}  # cvxopt 1.1.8
    solvers.options['msg_lev'] = 'GLP_MSG_OFF'  # cvxopt 1.1.7
    solvers.options['LPX_K_MSGLEV'] = 0  # previous versions

    #M = matrix(q).trans()
    M = matrix(q) #do not transpose
    n = M.size[1]

    A = np.hstack((np.ones((M.size[0], 1)), M))
    #Constraint: All P > 0
    eye_matrix = np.hstack((np.zeros((n, 1)), -np.eye(n)))

    A = np.vstack((A, eye_matrix))
    # Constraint: Sum(P) == 1
    A = matrix(np.vstack((A, np.hstack((0,np.ones(n))), np.hstack((0,-np.ones(n))))))

    #Create b Matrix
    b = matrix(np.hstack((np.zeros(A.size[0] - 2), [1, -1])))

    #Create C Matrix
    c = matrix(np.hstack(([-1], np.zeros(n))))

    sol = solvers.lp(c,A,b, solver=glpksolver)


    return sol['primal objective']
