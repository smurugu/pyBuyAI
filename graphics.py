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

def plot_rewards_per_episode(df,alpha=(0.7,1),colours=('tab:blue','tab:purple'),rolling_mean_window=10):
    df2 = df[df['bidding_round'] == max(df['bidding_round'])]
    df2['reward_ma'] = df2['reward'].rolling(rolling_mean_window,1).mean()

    fig, axs = plt.subplots(1)
    plt.plot(df2['episode'], df2['reward'], alpha=alpha[0], color='tab:blue', label='Reward')
    plt.plot(df2['episode'], df2['reward_ma'], alpha=alpha[1], color='tab:purple', label='{} game moving avg'.format(rolling_mean_window))
    plt.legend(loc='lower right')
    return (fig,axs)

if __name__ == '__main__':

    #get results paths graphics for a good agent
    results_folder = r'C:\GitRepo\pyBuyAI\parameter_grid_search2\results\epsilon_decay_gridsearch_10kg_4bp'
    game_id = '533e37b7-983c-4e45-93c7-7cb02d67048d'
    path_hdf = os.path.join(results_folder,'player_0_'+game_id+'.hdf')
    path_df = pd.read_csv(path_hdf,sep='#')
    path_df = path_df[path_df['episode']<3500]

    fig, axs = plot_rewards_per_episode(path_df, alpha=(0.5, 1), colours=('tab:blue', 'tab:purple'),
                                        rolling_mean_window=10)
    axs.set_title('Rewards per Episode')
    axs.set_xlabel('Episodes')
    axs.set_ylabel('Final Reward')

    plt.show()

    print('Graphics generation finish')