import pandas as pd
import json
import uuid
import os
import glob
import pickle

results_folder = r'./results'
results_file_template = os.path.join(results_folder,'*.pck')
results_files = glob.glob(results_file_template)

cols = {
    'player_id':'player_id',
    'agent_valuation': 'agent_valuation',
     'alpha': 'alpha',
     'bid_periods': 'bid_periods',
     'episodes': 'episodes',
     'epsilon': 'epsilon',
     'epsilon_decay_1': 'epsilon_decay_1',
     'epsilon_decay_2': 'epsilon_decay_2',
     'epsilon_threshold': 'epsilon_threshold',
     'gamma': 'gamma',
     'initial_state_random': 'initial_state_random',
     'num_players': 'num_players',
     'output_file': 'output_file',
     'output_folder': 'output_folder',
     'price_levels': 'price_levels',
     'q_convergence_threshold': 'q_convergence_threshold',
    }

all_results_df = pd.DataFrame()
for file in results_files:
    row_df = pd.DataFrame(columns=all_results_df.columns,index=[0])
    with open(file, 'rb') as handle:
        this_result = pickle.load(handle)

    # add config params into single df
    for col in cols.keys():
        if cols[col] in this_result.keys():
            row_df[col] = this_result[cols[col]]

    #now add learning results
    this_result_df = pd.DataFrame.from_dict(this_result['results_df'])
    if 'player_id' in this_result.keys():
        this_result_df = this_result_df[this_result_df['Player ID']==this_result['player_id']]
    for col in this_result_df.columns:
        row_df[col] = this_result_df[col]

    all_results_df = all_results_df.append(row_df)

all_results_df.to_csv(
    os.path.join(results_folder,'grid_search_results_{0}.hdf'.format(str(uuid.uuid4())))
    , index=False
    , sep='#')

print('Done aggregating results')