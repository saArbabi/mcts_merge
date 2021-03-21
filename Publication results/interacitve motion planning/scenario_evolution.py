import matplotlib.pyplot as plt
from importlib import reload
import numpy as np
import dill
# %%
"""
lead vehicle
"""
class Viewer():
    def __init__(self, env_config):
        self.env_config  = env_config
        self.fig_scene = plt.figure(figsize=(10, 9))
        self.fig_profiles = plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 14})
        self.fig_scene.subplots_adjust(left=None, bottom=0.15, right=None, \
                                top=None, wspace=None, hspace=0.3)
        self.fig_profiles.subplots_adjust(left=None, bottom=0.15, right=None, \
                                top=None, wspace=None, hspace=0.3)
        self.env_ax_1 = self.fig_scene.add_subplot(311, facecolor='lightgrey')
        self.env_ax_2 = self.fig_scene.add_subplot(312, facecolor='lightgrey')
        self.env_ax_3 = self.fig_scene.add_subplot(313, facecolor='lightgrey')
        self.v_ax = self.fig_profiles.add_subplot(311)
        self.along_ax = self.fig_profiles.add_subplot(312)
        self.alat_ax = self.fig_profiles.add_subplot(313)


    def draw_road(self, ax):
        lane_cor = self.env_config['lane_width']*self.env_config['lane_count']
        ax.hlines(0, 0, self.env_config['lane_length'], colors='k', linestyles='solid')
        ax.hlines(lane_cor, 0, self.env_config['lane_length'],
                            colors='k', linestyles='solid')

        if self.env_config['lane_count'] > 1:
            lane_cor = self.env_config['lane_width']
            for lane in range(self.env_config['lane_count']-1):
                ax.hlines(lane_cor, 320, 420,
                                colors='white', linestyles='--', linewidth=4)
                lane_cor += self.env_config['lane_width']
        ax.set_xlim(320, 400)
                # ax.plot(range(len(veh.x_track)), veh.y_track, color='red')
        ax.set_xlabel(' Longitudinal position ($m$)')
        ax.set_ylabel('Lateral position ($m$)')

    def draw_v_profile(self, ax, vehicles):
        for veh in vehicles:
            if veh.id == 'cae':
                ax.plot(self.time, veh.v_track, color='green')

            elif veh.id == 'm':
                ax.plot(self.time, veh.v_track, color='red')
                # print(veh.v_track)

            # print(str(veh.id), len(veh.v_track))
        ax.set_ylabel('Long. speed ($ms^{-1}$)')
        ax.set_xlabel('Time ($s$)')
        ax.set_xlim(0, 6)
        ax.yaxis.set_ticks(np.arange(10.5, 13, 0.5))

        ax.grid(alpha=0.8)

    def draw_along_profile(self, ax, vehicles):
        for veh in vehicles:
            if veh.id == 'cae':
                ax.plot(self.time, veh.along_track, color='green')
            elif veh.id == 'm':
                ax.plot(self.time, veh.along_track, color='red')
        # ax.set_xlim(0, 7)
        ax.yaxis.set_ticks(np.arange(-2, 2, 0.5))
        ax.set_xlim(0, 6)
        ax.grid(alpha=0.8)
        ax.set_ylabel('Long. acceleration ($ms^{-2}$)')
        ax.set_xlabel('Time ($s$)')

    def draw_alat_profile(self, ax, vehicles):
        for veh in vehicles:
            if veh.id == 'cae':
                ax.plot(self.time, veh.alat_track, color='green')
            elif veh.id == 'm':
                ax.plot(self.time, veh.alat_track, color='red')
        # ax.set_xlim(0, time[-1])
        ax.grid(alpha=0.8)
        ax.set_xlim(0, 6)
        ax.yaxis.set_ticks(np.arange(-0.5, 1.6, 0.5))
        ax.set_ylabel('Lateral speed ($ms^{-1}$)')
        ax.set_xlabel('Time ($s$)')

    def draw_xy_profile(self, axs, vehicles):
        chunk = 0
        chunk_step = int(len(vehicles[0].x_track)/3)
        for ax in axs:
            for veh in vehicles:
                veh_x = veh.x_track[0: chunk+chunk_step]
                veh_y = veh.y_track[0: chunk+chunk_step]

                if veh.id == 'cae':
                    ax.plot(veh_x, veh_y, color='green')
                elif veh.id == 'm':
                    ax.plot(veh_x, veh_y, color='red')
                elif veh.id == 'y':
                    ax.scatter(veh_x[-1], veh_y[-1], color='grey')
                else:
                    ax.scatter(veh_x[-1], veh_y[-1], color='grey')

            chunk += chunk_step

    def update_plots(self, vehicles):
        self.time = [0]
        for t in range(len(vehicles[0].x_track)-1):
            self.time.append(self.time[-1]+0.1)



        self.draw_road(self.env_ax_1)
        self.draw_road(self.env_ax_2)
        self.draw_road(self.env_ax_3)
        self.draw_xy_profile([self.env_ax_1,
                                self.env_ax_2,
                                self.env_ax_3],
                                vehicles)
        self.fig_scene.savefig("env_evolution.png", dpi=500, bbox_inches = 'tight',
        pad_inches = 0)
        # plt.close()
        self.draw_v_profile(self.v_ax, vehicles)
        self.draw_along_profile(self.along_ax, vehicles)
        self.draw_alat_profile(self.alat_ax, vehicles)
        self.fig_profiles.savefig("env_profiles.png", dpi=500, bbox_inches = 'tight',
        pad_inches = 0)
        # plt.close()
                # plt.show()

class Env():
    def __init__(self):
        self.viewer = None
        self.vehicles = [] # all vehicles
        self.default_config()
        self.set_stateIndex()

    def default_config(self):
        self.config = {'lane_count':3,
                        'lane_width':3.7, # m
                        'lane_length':100, # m
                        }


    def set_stateIndex(self):
        self.indx_m = {}
        self.indx_y = {}
        self.indx_f = {}
        self.indx_fadj = {}
        i = 0
        for name in ['vel', 'pc', 'act_long_p','act_lat_p']:
            self.indx_m[name] = i
            i += 1

        for name in ['vel', 'dx', 'act_long_p']:
            self.indx_y[name] = i
            i += 1

        for name in ['vel', 'dx', 'act_long_p']:
            self.indx_f[name] = i
            i += 1

        for name in ['vel', 'dx', 'act_long_p']:
            self.indx_fadj[name] = i
            i += 1

    def reset(self):
        obs_history = self.obs_module.scale_obs_history()
        action_conditional = self.get_conditional(obs_history)
        return obs_history, action_conditional

    def step(self, actions):
        for action in actions:
            for vehicle in self.vehicles:#
                if vehicle.id == 'cae':
                    vehicle.step(action)
                else:
                    vehicle.step()

            current_obs = self.obs_module.get_current_obs()
            new_obs = self.update_obs(action, current_obs)
            self.obs_module.update_obs_history(new_obs)

        obs_history = self.obs_module.scale_obs_history()
        action_conditional = self.get_conditional(obs_history)
        return obs_history, action_conditional

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.config)
        self.viewer.update_plots(self.vehicles)

    def get_conditional(self, obs_history):
        action_conditional = [obs_history[-1, self.indx_m['act_long_p']],
                            obs_history[-1, self.indx_m['act_lat_p']],
                            obs_history[-1, self.indx_y['act_long_p']],
                            obs_history[-1, self.indx_f['act_long_p']],
                            obs_history[-1, self.indx_fadj['act_long_p']]]
        return [np.array([[a]]) for a in action_conditional]

    def update_obs(self, action, current_obs):
        new_obs = current_obs.copy()

        new_obs[self.indx_y['dx']] = abs(self.veh_y.x-self.veh_cae.x)
        new_obs[self.indx_f['dx']] = abs(self.veh_f.x-self.veh_cae.x)
        new_obs[self.indx_fadj['dx']] = abs(self.veh_fadj.x-self.veh_cae.x)

        new_obs[self.indx_m['vel']] = self.veh_cae.v
        new_obs[self.indx_y['vel']] = self.veh_y.v
        new_obs[self.indx_f['vel']] = self.veh_f.v
        new_obs[self.indx_fadj['vel']] = self.veh_fadj.v

        new_obs[self.indx_m['act_long_p']] = action[0]
        new_obs[self.indx_m['act_lat_p']] = action[1]
        new_obs[self.indx_y['act_long_p']] = self.veh_y.a
        new_obs[self.indx_f['act_long_p']] = self.veh_f.a
        new_obs[self.indx_fadj['act_long_p']] = self.veh_fadj.a

        self.veh_cae.pc += action[1]*0.1
        if self.veh_cae.pc > 1.85:
            self.veh_cae.pc = -1.85
        elif self.veh_cae.pc < -1.85:
            self.veh_cae.pc = 1.85
        new_obs[self.indx_m['pc']] = self.veh_cae.pc
        # print('action: ', self.action_conditional)
        return new_obs

class Vehicle(object):
    STEP_SIZE = 0.1 # s update rate - used for dynamics
    def __init__(self, id, lane_id, x, v):
        self.v = v # longitudinal speed [m/s]
        self.lane_id = lane_id # inner most right is 1
        self.y = 0
        self.x = x # global coordinate
        self.id = id
        self.y = 2*lane_id*1.85-1.85

    def act(self):
        """
        :param high-lev decision of the car
        """
        pass

    def observe(self):
        raise NotImplementedError

    def act(self):
        raise NotImplementedError

    def step(self, action):
        self.x = self.x + self.v * self.STEP_SIZE \
                                    + 0.5 * action * self.STEP_SIZE **2

        self.v = self.v + action * self.STEP_SIZE

class HumanVehicle(Vehicle):
    def __init__(self, lane_id, x, state_arr, state_indx, id):
        self.state_arr = state_arr
        self.state_indx = state_indx
        self.env_step = 19 # because you need history
        self.x_track = []
        self.y_track = []
        self.along_track = []
        self.v_track = []
        super().__init__(id, lane_id, x, self.state_arr[self.env_step, self.state_indx['vel']])
        if id == 'm':
            self.alat_track = []

    def track_vals(self):
        if self.id == 'm':
            self.along_track.append(self.state_arr[self.env_step+1, self.state_indx['act_long_p']])
            self.alat_track.append(self.state_arr[self.env_step+1, self.state_indx['act_lat_p']])
            self.v_track.append(self.state_arr[self.env_step, self.state_indx['vel']])

        self.x_track.append(self.x)
        self.y_track.append(self.y)

    def step(self):
        self.track_vals()
        self.a = self.state_arr[self.env_step+1, self.state_indx['act_long_p']]
        self.v = self.state_arr[self.env_step, self.state_indx['vel']]
        self.x = self.x + self.v * self.STEP_SIZE \
                                    + 0.5 * self.a * self.STEP_SIZE **2

        if self.id == 'm':
            self.y += self.state_arr[self.env_step+1, self.state_indx['act_lat_p']] * self.STEP_SIZE
        self.env_step += 1

class CAEVehicle(Vehicle):
    def __init__(self, lane_id, x, v, pc, id='cae'):
        super().__init__(id, lane_id, x, v)
        self.obs_history = []
        self.x_track = []
        self.y_track = []
        self.v_track = []
        self.along_track = []
        self.alat_track = []
        self.pc = pc
        self.pc_track = [self.pc]

    def act(self, obs_history, action_conditional, bc):
        # if len(self.obs_history) % 30 == 0:
        executable_plan, bc = self.policy.mpc(obs_history, action_conditional, bc)
        return executable_plan, bc

    def track_vals(self, action):
        """what is the state/action in the current step?
        Note: used for plots
        """
        self.x_track.append(self.x)
        self.y_track.append(self.y)
        self.v_track.append(self.v)
        self.pc_track.append(self.pc)
        self.along_track.append(action[0])
        self.alat_track.append(action[1])

    def step(self, action):
        self.track_vals(action)
        self.x = self.x + self.v * self.STEP_SIZE \
                                    + 0.5 * action[0] * self.STEP_SIZE **2
        self.v = self.v + action[0] * self.STEP_SIZE
        y_delta = action[1]*self.STEP_SIZE
        self.y += y_delta

class ObservationModule():
    def __init__(self, state_arr, targ_arr, current_step, data_obj):
        self.state_arr = state_arr
        self.targ_arr = targ_arr
        self.obs_history = None # unscaled
        self.current_step = current_step # must be greater than 19 - history length
        self.start_step = current_step - 19
        self.data_obj = data_obj
        self.sequence_obs()

    def sequence_obs(self):
        self.obs_history = self.state_arr[self.start_step:self.current_step+1, :]

    def scale_obs_history(self):
        return self.data_obj.applyStateScaler(self.obs_history.copy())

    def get_current_obs(self):
        return self.obs_history[-1, :]

    def update_obs_history(self, new_obs):
        # print('testing::::::', self.obs_history.shape)

        obs_history = self.obs_history.tolist()
        obs_history.pop(0)
        obs_history.append(list(new_obs))
        # print('testing::::::', obs_history)

        self.obs_history = np.array(obs_history)

from models.core.train_eval.utils import loadConfig
import os
os.getcwd()
# os.chdir('../')
from planner import model_evaluator
reload(model_evaluator)
from planner import policy
reload(policy)
from planner import test_data_handler
reload(test_data_handler)
from planner.policy import MergePolicy
from planner.model_evaluator import ModelEvaluation
from planner.test_data_handler import TestdataObj


exp_to_evaluate = 'series077exp001'
config = loadConfig(exp_to_evaluate)
traffic_density = ''
test_data = TestdataObj(traffic_density, config)
model = MergePolicy(test_data, config)
eval_obj = ModelEvaluation(model, test_data, config)
st_seq, cond_seq, st_arr, targ_arr = eval_obj.episodeSetup(2215)
st_i, cond_i, bc_der_i, history_i, _, targ_i = eval_obj.sceneSetup(st_seq,
                                                cond_seq,
                                                st_arr,
                                                targ_arr,
                                                current_step=19,
                                                pred_h=2)
env = Env()
veh_cae = CAEVehicle(lane_id=2, x=320, v=st_arr[19, env.indx_m['vel']], pc=st_arr[19, env.indx_m['pc']], id='cae')
env.obs_module = ObservationModule(st_arr.copy(), targ_arr, 19, test_data.data_obj)
veh_cae.policy = model
veh_cae.policy.replanning_rate = 3 # every n steps, replan
###
veh_m = HumanVehicle(lane_id=2, x=320, state_arr=st_arr, state_indx=env.indx_m, id='m')
###
x = veh_m.x-st_arr[0, env.indx_y['dx']]
veh_y = HumanVehicle(lane_id=3, x=x, state_arr=st_arr, state_indx=env.indx_y, id='y')
###
x = veh_m.x+st_arr[0, env.indx_f['dx']]
veh_f = HumanVehicle(lane_id=2, x=x, state_arr=st_arr, state_indx=env.indx_f, id='f')
###
x = veh_m.x+st_arr[0, env.indx_fadj['dx']]
veh_fadj = HumanVehicle(lane_id=3, x=x, state_arr=st_arr, state_indx=env.indx_fadj, id='fadj')
###
env.vehicles = [veh_cae, veh_m, veh_y, veh_f, veh_fadj]
env.veh_cae = veh_cae
env.veh_y = veh_y
env.veh_f = veh_f
env.veh_fadj = veh_fadj
env.eval_obj = eval_obj
veh_m.x - veh_fadj.x
veh_m.x - veh_y.x
len(veh_m.x_track)
veh_y.x
obs_history, action_conditional = env.reset()
np.set_printoptions(suppress=True)
bc = bc_der_i
for i in range(18):
    executable_plan, bc = veh_cae.act(obs_history.copy(), action_conditional.copy(), bc.copy())
    obs_history, action_conditional = env.step(executable_plan)
env.render()

# plt.savefig("scene_evolution.png", dpi=500)


# %%
plt.plot(veh_f.vel_profile)
plt.plot(veh_f.vel_profile)
plt.plot(veh_m.vel_profile)
plt.plot(veh_cae.pc_track)

a = np.array([1.23412])
a
a.round(2)
# %%
a = np.array([[1,2,3],[4,5,6]])
list(a)
a.tolist()
a = [i ]
