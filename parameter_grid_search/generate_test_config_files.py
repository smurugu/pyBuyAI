import itertools
import uuid
import os

output_folder = r'./config_files'

config_template = """Echo Running Config: %0

SET ARG_STRING=^
episodes:XXX,^
initial_state_random:XXX,^
randomise_turn_order:XXX,^
share_rewards_on_tie:XXX,^
bid_periods:XXX,^
price_levels:XXX,^
num_players:XXX,^
alpha:XXX,^
gamma:XXX,^
epsilon:XXX,^
epsilon_decay_1:XXX,^
epsilon_decay_2:XXX,^
epsilon_threshold:XXX,^
q_update_mode:XXX,^
agent_valuation:XXX,^
q_convergence_threshold:XXX,^
output_folder:r'./results',^
output_file:%~n0

python.exe ..\__main__.py %ARG_STRING%"""

replace_dict = {
    'episodes':[40000],
    'initial_state_random':[False],
    'randomise_turn_order':[True,False],
    'share_rewards_on_tie':[True,False],
    'bid_periods':[3],
    'price_levels':[5],
    'num_players':[2],
    'alpha':[0.8],
    'gamma':[0.5],
    'epsilon':[1],
    'epsilon_decay_1':[0.99999],
    'epsilon_decay_2':[0.9],
    'epsilon_threshold':[0.4],
    'agent_valuation':[3.3],
    'q_update_mode':['foe','friend','qlearn'],
    'q_convergence_threshold':[100]
}
# format strings for replacement
replace_tuple = ()
for k in replace_dict.keys():
    vals = replace_dict[k]
    this_replace_list = [k+':'+str(v) for v in vals]
    replace_tuple = replace_tuple + ([k+':XXX',this_replace_list],)

#get all required combinations for string replace (cartesian, one from each list)
thing = []
for element in itertools.product(*[x[1] for x in replace_tuple]):
    thing = thing +[[(replace_tuple[i][0],v) for i,v in enumerate(element)]]

# check and create output folder
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

files_written = []
# do string replacements
for configuration in thing:
    this_cfg_string = config_template
    for item in configuration:
        this_cfg_string = this_cfg_string.replace(item[0],item[1])
    file_name = os.path.join(output_folder,'run_'+str(uuid.uuid4())+'.bat')
    f = open(file_name,'w+')
    f.write(this_cfg_string)
    f.close()
    files_written = files_written + ['CALL '+file_name]

for file in files_written:
    print(file)

print("lala I'm done")
