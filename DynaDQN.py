# Categorize Continuous State Space using Binning
# Aggregate reward in Q-Matrix using dictionary
# Hallucinate future actions using model T
# Hallucinate future rewards using alternative reward model R
# Update Q-Matrix with T and R to increase sample effeciency
# \\ \\ \\ \\
# Developed by Nathan Shepherd
# Inspired by Udemy tutorial @ https://bit.ly/2w3dSmr
# Sutton and Barto. Reinforcement Learning: An Introduction.
#    MIT Press, Cambridge, MA, 1998. 

import gym
import random
import operator
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

def max_dict(d):
    max_val = float('-inf')
    max_key = ""
    for key, val in d.items():
        if val > max_val:
            max_val = val
            max_key = key
    return max_key, max_val

class DQN:
    def __init__(self, obs_space, num_bins, num_actions, observation):

        self.obs_space = obs_space
        self.num_actions = num_actions
        
        self.bins = self.get_bins(num_bins)
        self.init_Q_matrix(observation)

        self.time_step = 0
        self.T = {}# probability of s' given (s, a)
        self.starts = {}# initial states of environment
        self.ends = {}# terminal states of environment
        
        self.R = {}# expected reward given (s, a)

    def init_Q_matrix(self, obs):
        assert(len(obs)==self.obs_space)
        states = []
        for i in range(10**(self.obs_space)):
            #populates state with left padded numbers as str
            states.append(str(i).zfill(self.obs_space))

        self.Q = {}
        for state in states:
            self.Q[state] = {}
            for action in range(self.num_actions):
                self.Q[state][action] = 0
    

    def get_action(self, state):
        #print(state)
        string_state = ''.join(str(int(elem)) for elem in self.digitize(state))
        return max_dict( self.Q[string_state] )[0]

    def evaluate_utility(self, state):
        string_state = ''.join(str(int(elem)) for elem in self.digitize(state))
        return max_dict( self.Q[string_state] )[1]

    def update_T(self, state, action, state_next, info):
        #Update T with <s, a, s'>
        # T[s][a][s'] += 1 #thus T is the probability of being in s' given (s, a)
        #   to evalute P[s' | (s, a)]
        #   --> max(T[s][a]) for all possible s'
        t_str = state + str(action)
        state_next = ''.join(str(int(elem)) for elem in self.digitize(state_next))
        if info['start']:
            if t_str not in self.starts:
                self.starts[t_str] = 1
            else: self.starts[t_str] += 1
        if info['end']:
            if t_str not in self.ends:
                self.ends[t_str] = 1
            else: self.ends[t_str] += 1
        try:
            self.T[t_str][state_next] += 1
        except KeyError:
            try: self.T[t_str][state_next] = 10**(-3)
            except KeyError: self.T[t_str] = {}

    def update_R(self, state, action, reward):
        #Update R with E[s, a]
        # R'[s, a] = (1 - ALPHA)R[s, a] + ALPHA*R
        #   or try alternative update
        if state not in self.R:
            self.R[state] = reward
        else: self.R[state] += reward - self.R[state]
        
    def update_policy(self, state, state_next, action, reward, info):
        state_value = self.evaluate_utility(state)
        
        action = self.get_action(state)
        reward_next = self.evaluate_utility(state_next)

        state_value += ALPHA*(reward + GAMMA * reward_next - state_value)


        state = ''.join(str(int(elem)) for elem in self.digitize(state))
        self.Q[state][action] = state_value


        self.update_T(state, action, state_next, info)
        self.update_R(state, action, state_value)


        self.time_step += 1
        if self.time_step % 14000 == 0:
            print('replaying from model')
            for i in range(200): self.simulate_experience()

    def get_next_state(self, init):
        probs = self.T[init]
        total = 0;
        for c in probs.values(): total += c
        for key in probs:
            probs[key] = (probs[key] / total)

        probs = sorted(probs.items(), key=operator.itemgetter(1))
##        print()
##        for p in probs:
##            print(p)
        #state_next = probs[-1][0]
        if random.random() < 0.8:
            p_distro = [val[1] for val in probs][::-1]
        else:
            p_distro = [val[1] for val in probs]

        if len(p_distro) == 0: raise KeyError
        state_next = np.random.choice([key[0] for key in probs], 1,
                                       p=p_distro)[0]
        
        t_str = state_next + str(max_dict( self.Q[state_next] )[0])
       # print('Next state and action:', t_str)
        return t_str
            
    def simulate_experience(self):
        #Use T and R to hallucinate future experiences and update Q-Matrix
        #print("Simulating experience and updating Q-Matrix")
        state, val = max_dict(self.starts)
        done = False; step = 0;

        transitions = []
        while state not in self.ends and step < 200:
            step += 1
            try:
                state_next = self.get_next_state(state)
            except KeyError: break
            

            reward = self.R[state[:-1]]

            s = state[:-1]; a = int(state[-1])
            self.Q[s][a] += (reward - self.Q[s][a])
            state = state_next
        #print('completed self training episode')

        
    def get_bins(self, num_bins):
        # Make 10 x state_depth matrix,  each column elem is range/10
        # Digitize using bins ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # obs[0]--> max: 0.265506 | min: -0.149958 | std: 0.151244
        # obs[1]--> max: 0.045574 | min: -0.036371 | std: 0.032354
        # obs[2]--> max: 0.241036 | min: -0.336625 | std: 0.205835
        # obs[3]--> max: 0.046279 | min: -0.051943 | std: 0.039247        
        
        bins = []
        ranges = [4.8, 5, 0.418, 5]
        for i in range(self.obs_space):
            # use minimum value to anchor buckets
            start, stop = -ranges[i], ranges[i]
            buckets = np.linspace(start, stop, num_bins)
            bins.append(buckets)

        return bins         
            
    def digitize(self, arr):
        # distrubute each elem in state to the index of the closest bin
        state = np.zeros(len(self.bins))
        for i in range(len(self.bins)):
            state[i] = np.digitize(arr[i], self.bins[i])
        return state


    def get_state_stats(self):
        for i in range(len(self.Q)):
            print("\nElem:",i,end=" ")
            keys = [key for key in self.Q[i].keys()]
            print("Range: [%s, %s]" % (min(keys), max(keys)),
                  "STDDEV:", round(np.std(keys), 3), "Count:" , len(keys))


def play_episode(agent, act_space, epsilon=.2, viz=False):
    state = env.reset()
    total_reward = 0
    terminal = False
    started = False
    num_frames = 0
    

    info = {'start':  True ,
            'end'  :  False }

    max_rwd = -200
    while not terminal:
        if viz: env.render()
        if started: info['start'] = False
        started = True

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = agent.get_action(state)
        
        state_next, reward, terminal,_ = env.step(action)

        total_reward += reward
        
        if terminal:
            #if num_frames > 150:
            #    reward += np.log(num_frames)
        
            if  num_frames < 200:
                reward = -300

            info['end'] = True

        action_next = agent.get_action(state_next)

        agent.update_policy(state, state_next, action, reward, info)
              
        state = state_next
        num_frames += 1
        
    
    return total_reward, num_frames

def train(obs_space, act_space=None,epochs=2000, obs=False, agent=False):
    if not agent: agent = DQN(obs_space, NUM_BINS, act_space, env.reset())

    stacked_frames = []
    rewards = [0]; avg_rwd = 0
    dr_dt = 0#reward derivitive with respect to time
    for ep in range(1, epochs):
        epsilon = max(EPSILON_MIN, np.tanh(-ep/(min(epochs, 2000)/2))+ 1)
                      

        ep_reward, num_frames = play_episode(agent, act_space, epsilon, viz=obs)
        if ep % 100 == 0:
            avg_rwd = round(np.mean(rewards),3)
            dr_dt = round(abs(dr_dt) - abs(avg_rwd), 2)
            print("Ep: {} | {}".format(ep, epochs),
                  "%:", round(ep*100/epochs, 2),
                  "Eps:", round(epsilon, 2),
                  "Avg rwd:", round(avg_rwd , 2),
                  "Ep rwd:", int(ep_reward),
                  "dr_dt:", -dr_dt)

        stacked_frames.append(num_frames)
        rewards.append(ep_reward)
        dr_dt = round(avg_rwd,2)

    return rewards, stacked_frames, agent

def observe(agent, N=15):
    [play_episode(agent, -1, viz=True) for ep in range(N)]

def plot_running_avg(reward_arr):
    N = len(reward_arr)
    running_avg = []

    step_size = 75
    for t in range(step_size, N):
        running_avg.append( np.mean(reward_arr[t-step_size: t+1]))
    x = [i for i in range(len(running_avg))]
    plt.plot(x, running_avg, color="purple", label="Q-Learning Running Average")
    plt.show()

def play_random(viz=False):
    observation = env.reset()
    total_reward = 0
    terminal = False

    while not terminal:
        if viz: env.render()
        action = env.action_space.sample()
        observation, reward, terminal, info = env.step(action)
        total_reward += reward

        #if terminal and num_frames < 200:
         #   reward = -300
        
    return total_reward

gym.envs.register(
    id='CartPoleExtraLong-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=250,
    reward_threshold=-110.0,
)
env = gym.make('CartPoleExtraLong-v0')
#env = gym.make('CartPole-v0')
observe_training = False
EPSILON_MIN = 0.1
NUM_BINS = 10
ALPHA = 0.01
GAMMA = 0.9
################################TODO: ADD reward for cart staying in center

EPOCHS = 2000#2000
'''
    StringDQN learns a policy using model-free reinforcement learning.
              The observation vector is digitized using binning and
              concatenated into a string.
    
    StringDQN reaches plateau of ~150 after  ~900 of 1500 episodes
              reaches plateau of ~250 after ~1500 of 2000 episodes

    DynaDQN uses StringDQN to sample a dynamic policy. The nature of
            the environment are computed using a statistical model T.
            The rewards associated with the 'Hallucinated' environment
            are updated with the off policy R = E[s,a]. This experience
            is then used to update the StringDQN policy periodically.
    
'''
obs_space = 4
action_space = env.action_space.n

if __name__ == "__main__":
    episode_rewards, _, Agent = train(obs_space, act_space=action_space,
                                      epochs = EPOCHS, obs = observe_training)
    
    #random_rwds = [play_random() for ep in range(EPOCHS)]

    plt.title("Average Reward with Q-Learning By Episode (CartPole)")
    plot_running_avg(episode_rewards)
    #plt.plot(random_rwds, color="gray", label="Random Moves Running Average")

    #plt.xlabel('Training Time (episodes)', fontsize=18)
    #plt.ylabel('Average Reward per Episode', fontsize=16)
    #plt.legend()
    























