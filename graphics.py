import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import environment as env
import agent

file_name = 'player0_T5_S11_a08_g05_e06065230778740716_ed1099995_ed20999.npy'

player = agent.Player().load_serialised_agent(file_name)

fig, axs = player.get_path_graphics(alpha=0.1,sub_plots=5)
plt.show()
fig.savefig(file_name.replace('.npy','.png'))
print('lala, end')