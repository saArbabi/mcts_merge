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
"""Compare rwse for different architectures and traffic densities
"""
discount_factor = 0.9
gamma = np.power(discount_factor, np.array(range(0,20)))

exps = [
        # 'series077exp001', # baseline
        # 'series078exp001', # only target car in conditional = to show interactions mater
        'series079exp002', # no teacher helping - to show it maters
        'series081exp001',
        ]
densities = ['low_density_','medium_density_', 'high_density_']

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
        'series079exp002',
        'series081exp001',

        ]
densities = ['high_density_']
densities = ['medium_density_']
# densities = ['low_density_']

discounted_exp_results = {}
exp_names = []
for exp in exps:
    for density in densities:
        exp_names.append(exp+density)

for exp_name in exp_names:
    discounted_exp_results[exp_name] = []

    for key in ['vel_m','lat_vel','vel_y','vel_fadj', 'vel_f']:
        discounted_exp_results[exp_name].append(np.sum(rwses[exp_name][key]*gamma))
# %%
"""To visualise rwse against prediction horizon
"""
densities = ['high_density_']
# densities = ['medium_density_']
# densities = ['low_density_']

discounted_exp_results = {}
exp_names = []
for exp in exps:
    for density in densities:
        exp_names.append(exp+density)

for key in ['vel_m','lat_vel','vel_y','vel_f','vel_fadj']:
    legends = []
    plt.figure()
    for exp_name in exp_names:
        plt.plot(rwses[exp_name][key])
        legends.append(key+'_'+exp_name)
    plt.legend(legends)
    plt.grid()
# %%
"""Bar chart visualistation
"""
labels = ['$\dot x_{0}$', '$\dot y_{0}$', '$\dot x_{1}$', '$\dot x_{2}$', '$\dot x_{3}$']
exp1 = discounted_exp_results[exp_names[0]]
exp2 = discounted_exp_results[exp_names[1]]
exp3 = discounted_exp_results[exp_names[2]]

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
""" rwse against training horizon
"""
densities = ['high_density_']

fig, axs = plt.subplots(1, 2, figsize=(8,3))
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
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('$\dot x_{0}$ RWSW [$ms^{-1}$] ')
axs[0].set_ylim([0,2.6])
axs[0].yaxis.set_ticks(np.arange(0, 2.6, 0.5))

axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('$\dot y_{0}$ RWSW [$ms^{-1}$] ')
axs[1].set_ylim([0,1.75])
axs[1].yaxis.set_ticks(np.arange(0, 1.76, 0.25))
legends = ['s=1','s=3','s=7', 's=10']
for key in range(2):
    for exp_name in exp_names:
        axs[key].plot(np.arange(0,2.1,0.1), rwses[exp_name][considered_states[key]])
    axs[key].grid(axis='y')
axs[0].legend(legends)

# %%

""" effect of step size
"""
fig, axs = plt.subplots(1, 2, figsize=(8,3))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

exps = [
        'series081exp002', # 0.3s
        'series082exp001', #
        'series082exp002', #
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
legends = ['$\Delta t=0.1s$', '$\Delta t=0.2s$','$\Delta t=0.3s$']
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('$\dot x_{0}$ RWSW [$ms^{-1}$] ')
axs[0].set_ylim([0,1.75])
# axs[0].yaxis.set_ticks(np.arange(0, 2.6, 0.25))

axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('$\dot y_{0}$ RWSW [$ms^{-1}$] ')
axs[1].set_ylim([0,1])
axs[1].yaxis.set_ticks(np.arange(0, 1.1, 0.25))


for key in range(2):
    for exp_name in exp_names:
        axs[key].plot(np.arange(0,2.1,0.1), rwses[exp_name][considered_states[key]])
    axs[key].grid(axis='y')
axs[0].legend(legends)
