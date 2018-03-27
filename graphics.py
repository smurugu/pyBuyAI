import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import environment as env
import agent
import os
import itertools
from numpy import nan
import glob

def rewards_graphics(player_list, episodes, bid_periods, price_levels, num_players):
    """
    Plot a graph showing how each players reward rate varied throughout the process
    """
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1, ])
    for p in player_list:
        ax.plot(p.rewards_vector, label='Player {p}'.format(p=p.player_id))
    ax.set_xlabel('Thousand Episodes')
    ax.set_ylabel('Mean Reward per Episode')
    ax.set_title('Mean Rewards per Episode')
    ax.legend()
    fig.savefig(env.get_environment_level_file_name(episodes, bid_periods, price_levels, num_players)+'.png')
    return fig, ax

def path_graphics(path_df,alpha=0.5,sub_plots=5,trial_intervals=None):

    first = path_df['episode'].min()
    last = path_df['episode'].max()
    #cannot plot nan actions: replace these with -1
    path_df['bid'] = path_df['bid'].fillna(-1)

    if trial_intervals is None:
        breaks = list(range(first, last, int(round((last - first) / sub_plots)))) + [last]
        trial_intervals = [(breaks[i], breaks[i + 1]) for i in range(len(breaks) - 1)]

    fig, axs = plt.subplots(len(trial_intervals), 1, figsize=(15, 15), sharex=True, sharey=True,
                            tight_layout=True)

    if len(path_df) == 0:
        logging.error('Agent.get_path_graphics : agent has empty path_df')
        return (fig,axs)

    for i,intv in enumerate(trial_intervals):
        if path_df[path_df['episode']==min(intv)]['episode'].count() > 0:
            eps = path_df[path_df['episode']==min(intv)].head(1)['epsilon'].values[0]
        else:
            eps = np.nan
        axs[i].set_title('Trials {0} to {1} using epsilon = {2}'.format(intv[0],intv[1],eps))
        axs[i].set_xlabel('Bid period')
        axs[i].set_ylabel('Bid Amount')
        for t in range(intv[0],intv[1]):
            axs[i].plot(path_df[path_df['episode']==t]['bidding_round'],path_df[path_df['episode']==t]['bid'],alpha=alpha)

    fig.tight_layout()

    return (fig,axs)

def plot_grid_search_heatmap(param1,param2,dependent_var,df):
    fig, axs = plt.subplots(1)
    df2 = df.pivot(param1,param2,dependent_var)
    sns.heatmap(df2,ax=axs)
    return (fig,axs)

def plot_final_bids_heatmap(bids_df,price_levels,title=''):
    bids_df['freq'] = 1
    prices = [-1] + list(range(price_levels))
    values = [list(bids_df[col].drop_duplicates()) for col in bids_df.columns]
    cartesian_values = list(itertools.product(prices, prices))
    cartesian_values = [(x) + (0,) for x in cartesian_values]
    df2 = pd.DataFrame(columns=bids_df.columns, data=cartesian_values)
    bids_df = pd.concat([bids_df,df2],axis=0)
    piv = bids_df.fillna(-1).pivot_table(index=bids_df.columns[0], columns=bids_df.columns[1], values='freq', aggfunc='sum')
    fig, axs = plt.subplots(1)
    axs.set_title(title)
    axs.xaxis.tick_top()
    sns.heatmap(piv, cmap="YlGnBu", ax=axs)
    return (fig,axs)

def plot_rewards_per_episode(df):
    df2 = df[df['bidding_round'] == max(df['bidding_round'])]

    fig, axs = plt.subplots(1)
    plt.plot(df2['episode'], df2['reward'],alpha=0.7)
    #plt.show()

    return (fig,axs)

if __name__ == '__main__':
    results_template = r"C:\GitRepo\pyBuyAI\parameter_grid_search\results\*results*csv"
    results_files = glob.glob(results_template)

    df = pd.read_csv(results_files[0])
    for file in results_files[1:]:
        df = pd.concat([df,pd.read_csv(file)],axis=0)

    value_col = 'avg_rwd_100g'
    df[value_col] = df['Avg Reward Vector'].apply(lambda x: x.split('_')[len(x.split('_'))-1])

    grid_params = ['alpha','gamma']
    df = df[grid_params+[value_col]].astype(float)
    piv = df.fillna(-1).pivot_table(
        index=grid_params[0],
        columns=grid_params[1],
        values=value_col,
        aggfunc='sum',
        fill_value=0
    )
    fig, axs = plt.subplots(1)
    axs.set_title('Grid search on: {}'.format(', '.join(grid_params)))
    axs.xaxis.tick_top()
    sns.heatmap(piv, cmap="Blues", ax=axs, linewidths=1)


    #get multiagent final bid heatmaps
    game_id = '67835b1b-3e53-4263-b4c6-479d06894375'
    folder = r"C:\GitRepo\pyBuyAI\parameter_grid_search\results"

    player0_serialised = os.path.join(folder,'player_0_'+game_id+'.npy')
    #player0_path = os.path.join(folder,'player_0_'+game_id+'.hdf')
    player1_serialised = os.path.join(folder,'player_1_'+game_id+'.npy')
    #player1_path = os.path.join(folder, 'player_1_' + game_id + '.hdf')

    player0 = agent.Player()
    player0.load_serialised_agent(player0_serialised)

    player1 = agent.Player()
    player1.load_serialised_agent(player1_serialised)
    path_dataframes = [player0.path_df,player1.path_df]

    max_bid=int(round(player0.path_df['bid'].max()))

    update_mode = player0.q_update_mode
    n_bids = int(100)
    final_bids_df = env.get_last_x_bids_array(path_dataframes,n_bids)
    title = '{}, last {} games'.format(update_mode, n_bids)
    fig,axs = plot_final_bids_heatmap(final_bids_df,max_bid,title)
    #fig.savefig(player.get_serialised_file_name() + '_final_{}bids_heatmap.png'.format(str(n_bids)))
    plt.show()

    #debug path values
    #look at final paths:
    player0.path_df.tail(10)

    #look at qmatrix and r matrix
    State = env.define_state()
    S = [State(current_winner=nan, current_bids=(nan, nan)),
         State(current_winner=1, current_bids=(nan, 0)),
         State(current_winner=1, current_bids=(nan, 1)),
         State(current_winner=1, current_bids=(nan, 2)),
         State(current_winner=1, current_bids=(nan, 3)),
         State(current_winner=1, current_bids=(nan, 4)),
         State(current_winner=0, current_bids=(0, nan)),
         State(current_winner=nan, current_bids=(0, 0)),
         State(current_winner=1, current_bids=(0, 1)),
         State(current_winner=1, current_bids=(0, 2)),
         State(current_winner=1, current_bids=(0, 3)),
         State(current_winner=1, current_bids=(0, 4)),
         State(current_winner=0, current_bids=(1, nan)),
         State(current_winner=0, current_bids=(1, 0)),
         State(current_winner=nan, current_bids=(1, 1)),
         State(current_winner=1, current_bids=(1, 2)),
         State(current_winner=1, current_bids=(1, 3)),
         State(current_winner=1, current_bids=(1, 4)),
         State(current_winner=0, current_bids=(2, nan)),
         State(current_winner=0, current_bids=(2, 0)),
         State(current_winner=0, current_bids=(2, 1)),
         State(current_winner=nan, current_bids=(2, 2)),
         State(current_winner=1, current_bids=(2, 3)),
         State(current_winner=1, current_bids=(2, 4)),
         State(current_winner=0, current_bids=(3, nan)),
         State(current_winner=0, current_bids=(3, 0)),
         State(current_winner=0, current_bids=(3, 1)),
         State(current_winner=0, current_bids=(3, 2)),
         State(current_winner=nan, current_bids=(3, 3)),
         State(current_winner=1, current_bids=(3, 4)),
         State(current_winner=0, current_bids=(4, nan)),
         State(current_winner=0, current_bids=(4, 0)),
         State(current_winner=0, current_bids=(4, 1)),
         State(current_winner=0, current_bids=(4, 2)),
         State(current_winner=0, current_bids=(4, 3)),
         State(current_winner=nan, current_bids=(4, 4))]
    cb = [s.current_bids for s in S]
    pd.options.display.max_columns = 40
    thing = pd.DataFrame(data=player1.Q[0], columns=cb, index=cb)
    thing

    print('done')
    """
    ept = 0.6
    df2 = df[(df['bid_periods'] == 5) & (df['epsilon_threshold'] == ept)]

    fig,axs=plot_grid_search_heatmap('epsilon_decay_1','epsilon_decay_2','Period Converged',df2)
    fig.suptitle('Epsilon Threshold: ept')

    fig, axs = plt.subplots(1, 3, figsize=(15, 15), sharex=True, sharey=True,
                            tight_layout=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])

    for i,th in enumerate([0.2,0.4,0.6]):
        df2 = df[(df['bid_periods'] == 5) & (df['epsilon_threshold'] == th)]
        df3 = df2.pivot('epsilon_decay_1','epsilon_decay_2','Period Converged')
        im = sns.heatmap(df3,ax=axs[i],cbar=i == 0,cbar_ax=None if i else cbar_ax)
        axs[i].set_title('Epsilon threshold: {0}'.format(th))

    fig.suptitle('Episodes until convergence for epsilon decay configurations')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig = plt.figure()
    plt.plot(df2['episode'], df2['reward'])
    plt.show()
    """



    print('lala, end')