
import os
os.getcwd()
# os.chdir('../')


from models.core.train_eval.utils import loadConfig
import matplotlib.pyplot as plt
from importlib import reload
import numpy as np
from planner import policy
reload(policy)
from planner.policy import TestdataObj, MergePolicy, ModelEvaluation
import dill

exp_to_evaluate = 'series081exp003'
config = loadConfig(exp_to_evaluate)
traffic_density = ''
# traffic_density = 'high_densit_'
# traffic_density = 'medium_density_'
# traffic_density = 'low_density_'
test_data = TestdataObj(traffic_density, config)

model = MergePolicy(test_data, config)
eval_obj = ModelEvaluation(model, test_data, config)
# eval_obj.compute_rwse(traffic_density)

# %%
"""get mean rwse for different traffic densityties.
"""
exps = [
        'series077exp001', # baseline
        'series078exp001', # only target car in conditional = to show interactions mater
        'series081exp003',
        'series081exp001',
        'series081exp004',
        'series081exp002',
        'series083exp002'
        ]
densities = ['medium_density_', 'high_density_']

rwses = {}
for exp_i in range(len(exps)):
    for density_i in range(len(densities)):
        dirName = './models/experiments/'+exps[exp_i]+'/'+densities[density_i]+'rwse'
        with open(dirName, 'rb') as f:
            rwses[exps[exp_i]+densities[density_i]] = dill.load(f, ignore=True)

# %%
exps = [
        # 'series077exp001',
        # 'series078exp001',
        'series081exp003',
        'series081exp001',
        'series081exp004',
        'series081exp002',
        'series083exp002'

        ]
densities = ['high_density_']
# densities = ['medium_density_']
# densities = ['low_density_']

mean_rwse = {}
exp_names = []
for exp in exps:
    for density in densities:
        exp_names.append(exp+density)

for exp_name in exp_names:
    mean_rwse[exp_name] = []

    for key in ['vel_m','lat_vel','vel_y','vel_fadj', 'vel_f']:
        mean_rwse[exp_name].append(np.mean(rwses[exp_name][key][0:20]))
mean_rwse
# %%
"""To visualise rwse against prediction horizon
"""
densities = ['high_density_']
# densities = ['medium_density_']
# densities = ['low_density_']

mean_rwse = {}
exp_names = []
for exp in exps:
    for density in densities:
        exp_names.append(exp+density)

for key in ['vel_m','lat_vel','vel_y','vel_f','vel_fadj']:
    legends = []
    plt.figure()
    for exp_name in exp_names:
        plt.plot(rwses[exp_name][key][0:20])
        legends.append(key+'_'+exp_name)
    plt.legend(legends)
    plt.grid()
# %%
"""Bar chart visualistation
"""
labels = ['$\dot x_{v0}$', '$\dot y_{v0}$', '$\dot x_{1}$', '$\dot x_{2}$', '$\dot x_{3}$']
exp1 = mean_rwse[exp_names[0]]
exp2 = mean_rwse[exp_names[1]]
exp3 = mean_rwse[exp_names[2]]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, exp1, width,
                                color='lightgrey', edgecolor='black', hatch='//')
rects2 = ax.bar(x + width, exp2, width,
                                color='grey', edgecolor='black')
rects2 = ax.bar(x - width, exp3, width,
                                color='grey', edgecolor='black', hatch='//')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time discounted RWSE')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid(axis='y', alpha=0.3)


fig.tight_layout()

plt.show()
fig.savefig("low_density_performance.png", dpi=200)

# %%

from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
plt.rcParams.update({'font.size': 14})
""" rwse against training horizon
"""
densities = ['high_density_']

fig, axs = plt.subplots(1, 2, figsize=(9,4))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

considered_states = ['vel_m','lat_vel']
""" effect of trainign horizon
"""
exps = [
        'series077exp008', # - 1 steps
        'series077exp005', # - 3 steps
        'series077exp001', # - 7 steps
        'series077exp004', # - 10 steps
        ]

rwses = {}
for exp_i in range(len(exps)):
    for density_i in range(len(densities)):
        dirName = './models/experiments/'+exps[exp_i]+'/'+densities[density_i]+'rwse'
        with open(dirName, 'rb') as f:
            rwses[exps[exp_i]+densities[density_i]] = dill.load(f, ignore=True)

exp_names = []
for exp in exps:
    for density in densities:
        exp_names.append(exp+density)
axs[0].set_xlabel('Time horizon (s)')
axs[0].set_ylabel('$\dot x_{v0}$ RWSE [$ms^{-1}$] ')
axs[0].set_ylim([0,2.6])
axs[0].yaxis.set_ticks(np.arange(0, 2.6, 0.5))

axs[1].set_xlabel('Time horizon (s)')
axs[1].set_ylabel('$\dot y_{v0}$ RWSE [$ms^{-1}$] ')
axs[1].set_ylim([0,1.75])
axs[1].yaxis.set_ticks(np.arange(0, 1.76, 0.25))
legends = ['$N=1$','$N=3$','$N=7$', '$N=10$']
for key in range(2):
    for exp_name in exp_names:
        axs[key].plot(np.arange(0,2.1,0.1), rwses[exp_name][considered_states[key]])
    axs[key].grid(axis='y')
axs[0].legend(legends)
plt.tight_layout()
plt.savefig("horizon_effect.png", dpi=500)

# %%

""" effect of step size
"""
fig, axs = plt.subplots(1, 2, figsize=(9,4))
fig.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=0.3, hspace=None)


exps = [
        'series082exp002', #
        'series082exp001', #
        'series081exp002', # 0.3s
        ]
rwses = {}
for exp_i in range(len(exps)):
    for density_i in range(len(densities)):
        dirName = './models/experiments/'+exps[exp_i]+'/'+densities[density_i]+'rwse'
        with open(dirName, 'rb') as f:
            rwses[exps[exp_i]+densities[density_i]] = dill.load(f, ignore=True)

exp_names = []
for exp in exps:
    for density in densities:
        exp_names.append(exp+density)
legends = ['$\delta t=0.1s$', '$\delta t=0.2s$','$\delta t=0.3s$']
axs[0].set_xlabel('Time horizon (s)')
axs[0].set_ylabel('$\dot x_{v0}$ RWSE [$ms^{-1}$] ')
axs[0].set_ylim([0,1.75])
# axs[0].yaxis.set_ticks(np.arange(0, 2.6, 0.25))

axs[1].set_xlabel('Time horizon (s)')
axs[1].set_ylabel('$\dot y_{v0}$ RWSE [$ms^{-1}$] ')
axs[1].set_ylim([0,1])
axs[1].yaxis.set_ticks(np.arange(0, 1.1, 0.25))


for key in range(2):
    for exp_name in exp_names:
        axs[key].plot(np.arange(0,2.1,0.1), rwses[exp_name][considered_states[key]])
    axs[key].grid(axis='y')
axs[0].legend(legends)
plt.tight_layout()
plt.savefig("step_size_effect.png", dpi=500)
