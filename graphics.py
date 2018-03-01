import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import environment as env
import agent

file_name = 'player0_T4_S11_a08_g05_eth04_ed1-09999_ed2-099.npy'
player = agent.Player().load_serialised_agent(file_name)
#player.path_df = player.get_path_log_from_hdf(player.get_serialised_file_name()+'.hdf')

fig, axs = player.get_path_graphics(alpha=0.03,sub_plots=5)
plt.show()
fig.savefig(file_name.replace('.npy','.png'))
print('lala, end')