import numpy as np
import pandas as pd
import logging
import random
import os
import environment as env

class Player(object):
    """
    Class represents learning agent
    Can learn using standard Q-learning, Friend or Foe multiagent learning
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
        #self.R = None
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
            r = 0
        else:
            if is_tie:
                if self.share_rewards_on_tie:
                    r = (agent_valuation - price_paid) / 2
                else:
                    r = 0
            else:
                r = (agent_valuation - price_paid)
        logging.info('Calculate reward: won_action={}, share_rewards_on_tie={}, is_tie={}, agent_valuation={}, price_paid={} -> Reward = {}'.format(
            won_auction, self.share_rewards_on_tie, is_tie, agent_valuation, price_paid, r
        ))
        return r

    def get_reward(self,a,is_final_period:bool):
        """
        Method calculates reward for action
        Note: since variable 'a' refers to the state resulting from action, we do not need to know the preceding state
        :param a:
        :param is_final_period:
        :return: reward for current period
        """

        bids = self.S[a].current_bids
        price_paid1 = bids[self.player_id]
        price_paid2 = bids[1-self.player_id]

        # if bidding has ended, current player has highest bid (setting nans to -1) and current player bid is not nan
        won_auction = is_final_period & (~np.isnan(price_paid1)) & ((price_paid1 >= price_paid2) | np.isnan(price_paid2))
        is_tie=env.get_winner((price_paid1,price_paid2))
        r = self.calc_final_reward(won_auction,price_paid1,self.agent_valuation,is_tie)

        return r

    def get_possible_bids(self):
        """
        Function gets sorted list of all possible bids implied by list of states
        :return: list
        """
        action_tuples = [s.current_bids for s in self.S]
        actions = sorted(list(set([cb[self.player_id] for cb in action_tuples])))
        return actions

    def get_q(self, S, bid_periods):
        """
        Method generates blank Q-matrix using format required for multi-agent implementation
        Dimensions: t,S,a1,a2
        :param S: list of possible states
        :param bid_periods: total number of bid periods per auction
        :return:
        """
        #set up a numpy array, dimensions [t,s,a,a]
        actions = self.get_possible_bids()
        Q = np.zeros((bid_periods,len(S),len(actions),len(actions)))
        return Q

    def set_q(self, S, bid_periods):
        """
        Method initialises a blank Q matrix of the correct dimensions
        :param S: list of possible states
        :param bid_periods: total number of bid periods per auction
        :return:
        """
        self.Q = self.get_q(S, bid_periods)
        return self.Q

    def get_available_actions(self,t,s):
        """
        Function returns a list of actions currently available to the agent
        Since transition function is deterministic, function directly returns the state resulting from action
        :param t: current time period
        :param s: current state
        :return: list of indices of states resulting from possible actions
        """
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
        Method selects an action according to the current epsilon value
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

    def select_action_greedy(self,t,s):
        """
        Method selects an action according to a greedy policy
        Assuming the other agent sticks with their current bid, agent selects the action with the highest Q-value
        Where multiple actions share a Q-value, select one randomly

        Note: since transition function is deterministic, method returns index of state corresponding to action choice

        :param t: current time period
        :param s: current state
        :return: index of state resulting from selected action
        """
        all_bids = self.get_possible_bids()
        qpayoff_matrix = self.Q[t,s]

        other_player_current_bid=self.S[s].current_bids[1-self.player_id]
        other_player_current_bid_index=all_bids.index(other_player_current_bid)

        #assuming other player will stick with current bid, select actions with highest q values
        q_payoff = qpayoff_matrix[:,other_player_current_bid_index]
        max_q = np.nanmax(q_payoff)
        max_locs = [x[0] for x in np.argwhere(q_payoff == max_q).tolist()]
        qv_max = [(a, max_q) for a in max_locs]
        highest_valued_new_bids = [all_bids[q[0]] for q in qv_max]
        new_bid_qvalues = [q[1] for q in qv_max]

        #Get resulting states for action choices corresponding to highest Q values
        current_state = self.S[s]
        current_bids = current_state.current_bids
        corresponding_bid_statuses = [tuple(h if i ==self.player_id else b for i,b in enumerate(current_bids)) for h in highest_valued_new_bids]
        corresponding_new_states = [current_state._replace(current_bids=cb) for cb in corresponding_bid_statuses]
        corresponding_new_states = [s._replace(current_winner=env.get_winner_for_state(s)) for s in corresponding_new_states]
        corresponding_new_state_indices = [self.S.index(ns) for ns in corresponding_new_states]

        a = random.choice(corresponding_new_state_indices)

        #make summary for log
        qv_summary = dict(zip(corresponding_new_states,new_bid_qvalues))
        logging.info('Player {0}: Highest-valued possible actions are: {1}'.format(self.player_id, qv_summary))
        logging.info('Player {0}: Action {1} selected at random from highest-valued list'.format(self.player_id, {a: self.state_dict[a]}))
        return a

    def select_action_maximin(self,t,s):
        """
        Not used: method selects actions according to a maximin policy.
        :param t:
        :param s:
        :return:
        """
        qpayoff_matrix = self.Q[t,s]

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
        """
        Method selects actions according to an exploratory policy
        Actions are selected according to a uniform distribution over possible actions
        :param t: current time
        :param s: current state
        :return:
        """
        actions = self.get_available_actions(t, s)
        a = np.random.choice(actions)
        logging.info('Action {0} selected at random.'.format({a:self.state_dict[a]}))
        return a

    def update_q(self,t,s,actions_taken,is_final_period:bool):

        self.update_q_value(t, actions_taken, self.q_update_mode, is_final_period)

    def calc_value_foe(self,payoff_matrix):
        """
        Calculates the Value of the next state under Friends learning, ie assuming all other players using maximin strategy
        :param payoff_matrix:
        :return:
        """
        return solve_maximin(payoff_matrix)[1]

    def calc_value_friend(self,payoff_matrix):
        """
        Calculates the Value of the next state under Friends learning, ie assuming all players work to
        maximise this player's return (max over whole payoff matrix)
        :param payoff_matrix:
        :return:
        """
        return np.max(payoff_matrix)

    def calc_value_qlearning(self,payoff_matrix,a2):
        """
        Calculates the Value of the next state under Q-learning, ie assuming the environment is fixed
        (other player's action cannot change)
        :param payoff_matrix:
        :param a2: bid of other player
        :return: V
        """
        return np.max(payoff_matrix[:,a2])

    def update_q_value(self,t,actions_taken,learning_type,is_final_period:bool):
        """
        Agent updates its Q matrix according to a q-learning, friends or foe rule
        ie the value of the next state is assumed to result from a minimax action by other players
        :param learning_type: accepts values 'foe', 'friend', 'qlearn'
        """
        # determine bids placed by current player and other player
        # a1 and price_paid1 refer to the CURRENT AGENT's action
        # a2 and price_paid2 refer to the OTHER AGENT's action
        possible_bids = self.get_possible_bids()
        a = actions_taken[self.player_id]['action']
        a1 = a
        price_paid1 = self.S[a1].current_bids[self.player_id]
        a1 = possible_bids.index(price_paid1)
        a2 = actions_taken[1-self.player_id]['action']
        price_paid2 = self.S[a2].current_bids[1-self.player_id]
        a2 = possible_bids.index(price_paid2)

        #Take state from dict populated by main script: this is to enable random turn-taking
        s = actions_taken[self.player_id]['state']

        Qold = self.Q[t,s,a1,a2]
        r = self.get_reward(a, is_final_period)

        logging.info('Reward for player {0} in time period {1} for action {2} from state {3} = {4}'.format(
            self.player_id, t, a1, s, r))

        Q_curr = self.Q[t,s,:,:]
        Q_curr = np.reshape(Q_curr,(len(possible_bids),len(possible_bids)))
        logging.debug(
            'Player {}: Q matrix for period {} from state {}: \n {}'.format(self.player_id,t,self.S[s],Q_curr))

        t2 = t + 1 if not is_final_period else t
        final_action = [actions_taken[x]['action'] for x in actions_taken if actions_taken[x]['order'] == 1][0]
        s2 = final_action
        #s2 = actions_taken[-1:][0]
        Q_next = self.Q[t2,s2,:,:]
        Q_next = np.reshape(Q_next,(len(possible_bids),len(possible_bids)))

        if learning_type == 'qlearn':
            V_next = self.calc_value_qlearning(Q_next,a2)
        elif learning_type == 'foe':
            V_next = self.calc_value_foe(Q_next)
        elif learning_type == 'friend':
            V_next = self.calc_value_friend(Q_next)
        else:
            logging.error("Invalid learning type: '{}'".format(learning_type))

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

        Q = self.Q
        Q[t,s,a1,a2] = Qnew
        #Q[t, s, a2, a1] = Qnew #try flipping dimensions
        self.Q = Q
        #logging.debug('Updated Q matrix: \n {0}'.format(self.Q))
        return self.Q

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

    def get_path_log_entry(self, episode, bidding_round, prev_state_index, action_index, is_final_period):
        #'episode','bidding_round','prev_state_index','prev_state_label','action_index','bid','alpha','gamma','epsilon','reward'
        row_df = pd.DataFrame(index=[0],columns=self.path_df.columns)
        for col in ['episode','bidding_round','prev_state_index','action_index']:
            row_df[col] = locals()[col]

        for col in ['alpha','gamma','epsilon','epsilon_decay_1','epsilon_decay_2','epsilon_threshold','agent_valuation']:
            row_df[col] = self.__getattribute__(col)

        row_df['prev_state_label'] = str(self.S[prev_state_index])
        row_df['bid'] = self.S[action_index].current_bids[self.player_id]
        row_df['reward'] = self.get_reward(action_index, is_final_period)
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

    def get_path_log_from_hdf(self,hdf_file):

        return pd.read_csv(hdf_file,sep='#')

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
    retaliatory_action = np.argmax(q[:,expected_enemy_action])
    value = q[retaliatory_action,expected_enemy_action]
    return (retaliatory_action,value)
