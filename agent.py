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
from math import ceil

class Player(object):
    """
    This is an agent
    Default values 0, 0, 0, 0, 0, 0, 0, 0, []
    """
    def __init__(self, player_id=0, alpha=0, gamma=0, epsilon=0, epsilon_decay_1=0, epsilon_decay_2=0, epsilon_threshold=0, agent_valuation=0, S=0, stationaryQ_episodes=0):
        self.player_id = player_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_1 = epsilon_decay_1
        self.epsilon_decay_2 = epsilon_decay_2
        self.epsilon_threshold = epsilon_threshold
        self.agent_valuation = agent_valuation
        self.S = S
        self.Q = None
        self.R = None
        self.path_df = pd.DataFrame(columns=['episode','bidding_round','bid','prev_state_index','prev_state_label','action_index','alpha','gamma','epsilon','reward'])
        if type(S) == list:
            self.state_dict = dict(zip(list(range(len(S))), S))
        self.stationaryQ_episodes = stationaryQ_episodes
        self.Q_converged = None
        self.rewards_vector = None

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
        return r

    def update_q(self,t,s,a,is_final_period:bool):
        """
        Agent updates its Q matrix
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
        a2 = self.select_action_greedy(t2,s2)
        Qnext = self.Q[t2,s2,a2]

        # 6) Use the Q-learning rule to calcualte the new Q(s,a) value and update the Q matrix accordingly
        Qnew = round(Qold + self.alpha*(r + self.gamma*Qnext - Qold),2)
        logging.info('Q({0},{1}) = {2} using alpha = {3} and gamma = {4}'.format(s,a,Qnew,self.alpha,self.gamma))

        Q = self.Q
        Q[t,s,a] = Qnew
        self.Q = Q
        logging.debug('Updated Q matrix: \n {0}'.format(self.Q))
        return self.Q

    def add_to_stationaryQ_episodes(self):
        self.stationaryQ_episodes += 1
        return self.stationaryQ_episodes

    def reset_stationaryQ_episodes(self):
        self.stationaryQ_episodes = 0
        return self.stationaryQ_episodes

    def set_Q_converged(self, episode):
        self.Q_converged = episode
        return self.Q_converged

    def set_rewards_vector(self, episodes):
        rewards_vector = []
        for k in range(0,int(ceil(episodes/1000))):
            start_idx = int(k*1000)
            end_idx = int((k+1)*1000)
            rewards_vector.append(sum(self.path_df.iloc[start_idx:end_idx]['reward'])/1000)
        self.rewards_vector = rewards_vector
        return self.rewards_vector

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

        for col in ['alpha','gamma','epsilon']:
            row_df[col] = self.__getattribute__(col)

        row_df['prev_state_label'] = str(self.S[prev_state_index])
        row_df['bid'] = self.S[action_index].current_bids[self.player_id]
        row_df['reward'] = self.get_reward(bidding_round, prev_state_index, action_index)

        return row_df

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

    def print_path_log(self, csv_path):
        try:
            self.path_df.to_csv(csv_path,index=False)
            return True
        except Exception as ex:
            logging.error('Unable to print csv: {0} \n Trying again with random filename'.format(csv_path,ex))
            csv_path = csv_path.replace('.csv','_file-id-'+str(uuid.uuid4().fields[2])+'.csv')
            self.path_df.to_csv(csv_path, index=False)
            return False

    def get_path_log_from_hdf(self,hdf_file):

        return pd.read_csv(hdf_file,sep='#')

    def get_path_graphics(self,alpha=0.5,sub_plots=5,trial_intervals=None):

        df = self.path_df

        first = df['episode'].min()
        last = df['episode'].max()
        #cannot plot nan actions: replace these with -1
        df['bid'] = df['bid'].fillna(-1)

        if trial_intervals is None:
            breaks = list(range(first, last, int(round((last - first) / sub_plots)))) + [last]
            trial_intervals = [(breaks[i], breaks[i + 1]) for i in range(len(breaks) - 1)]

        fig, axs = plt.subplots(len(trial_intervals), 1, figsize=(15, 15), sharex=True, sharey=True,
                                tight_layout=True)

        if len(df) == 0:
            logging.error('Agent.get_path_graphics : agent has empty path_df')
            return (fig,axs)

        for i,intv in enumerate(trial_intervals):
            if df[df['episode']==min(intv)]['episode'].count() > 0:
                eps = df[df['episode']==min(intv)].head(1)['epsilon'].values[0]
            else:
                eps = np.nan
            axs[i].set_title('Trials {0} to {1} using epsilon = {2}'.format(intv[0],intv[1],eps))
            axs[i].set_xlabel('Bid period')
            axs[i].set_ylabel('Bid Amount')
            for t in range(intv[0],intv[1]):
                axs[i].plot(df[df['episode']==t]['bidding_round'],df[df['episode']==t]['bid'],alpha=alpha)

        fig.tight_layout()

        return (fig,axs)


    def get_serialised_file_name(self):
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
        return file_name

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

