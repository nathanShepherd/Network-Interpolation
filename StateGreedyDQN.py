# Categorize Continuous State Space using Binning
# Aggregate reward in Q-Matrix using dictionary
# \\ \\ \\ \\
# Developed by Nathan Shepherd
# Inspired by Phil Tabor
# @ https://github.com/MachineLearningLab-AI/OpenAI-Cartpole

import gym
import pickle
import random
import operator
import numpy as np
import matplotlib.pyplot as plt

''' Note:
DQN is extremely sensitive to internal parameters
Improve convergance by solving subproblems with reward function
and picking a good set of range values for bins
'''

observe_training = False
EPSILON_MIN = 0.01
NUM_BINS = 8#must be even#
ALPHA = np.tanh
GAMMA = 0.9

EPOCHS = 10000

def max_dict(d):
    max_val = float('-inf')
    max_key = ""
    for key, val in d.items():
        if val > max_val:
            max_val = val
            max_key = key
    return max_key, max_val

def min_dict(d):
    min_val = float('inf')
    min_key = ""
    for key, val in d.items():
        if val < min_val:
            min_val = val
            min_key = key
    return min_key, min_val

class DQN:
    def __init__(self, obs_space, num_actions, observation, bin_ranges=None):

        self.obs_space = obs_space
        self.num_actions = num_actions

        self.T = {}# probability of s' given (s, a)
        self.bin_ranges = bin_ranges
        self.bins = self.get_bins()
        self.init_Q_matrix(observation)

        self.Policy = {'actions':[0],# Sequence of actions leading to greatest rwd
                       'states': [],
                       'rewards':[float('-inf')]}

    def init_Q_matrix(self, obs):
        assert(len(obs)==self.obs_space)
        self.Q = {}
        self.find(''.join(str(int(elem)) for elem in self.digitize(obs)))

    def find(self, state_string):
        try:
            self.Q[state_string]
        except KeyError as e:
            self.T[state_string] = {i:0 for i in range(self.num_actions)}
            self.Q[state_string] = {i:0 for i in range(self.num_actions)}

    def get_action(self, state, use_policy=True):        
        self.find(state)
        if use_policy:
            return max_dict( self.Q[state] )[0]
        else:
            return random.randint(0, self.num_actions - 1)

    def update_policy(self, state, state_next, action, reward):
        str_state = ''.join(str(int(elem)) for elem in self.digitize(state))
        state_next = ''.join(str(int(elem)) for elem in self.digitize(state_next))

        state_value = self.evaluate_utility(str_state)
        reward_next = self.evaluate_utility(state_next)

        self.update_T(str_state, action)
        action = self.get_action(str_state)

        state_value += ALPHA(reward + GAMMA * reward_next - state_value)

        self.Q[str_state][action] = state_value


    def update_T(self, state, action):
        #T is the probability of being in s' given (s, a)
        #Update T with <s, a, s'>
        # T[s][a][s'] += 1 
        #   to evalute P[s' | (s, a)]
        #   --> max(T[s][a]) for all possible s'
        self.T[state][action] += 1

    def sample_from_T(self, state):
        probs = self.T[state]; total = 0
        for c in probs.values(): total += c
        probs = sorted(probs.items(), key=operator.itemgetter(1))
        
        if total == 0: total = 0.1
        # Reverse probs -> low-freq states have higher prob
        p_distro = [val[1]/total for val in probs][::-1]
        if sum(p_distro) == 0 or 1 in p_distro:
            p_distro = [1/self.num_actions for i in range(self.num_actions)]

        action =  np.random.choice([k[0] for k in probs], 1, p=p_distro)
        #print(probs, p_distro, action)
        # from probability distribution 
        
        for a in self.T[state]:
            if self.T[state][a] < 2 :# == 0
                #Prioratize actions which get agent to unvisited s'
                return int(a)

        #action = random.randint(0, self.num_actions - 1)
        return int(action)
            

    def evaluate_utility(self, state):
        self.find(state)
        return max_dict( self.Q[state] )[1]


    def get_bins(self):
        # Make 10 x state_depth matrix,  each column elem is range/10
        # Digitize using bins ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ranges = [rng[0] for rng in self.bin_ranges]
        num_bins = [num[1] for num in self.bin_ranges]

        bins = []
        for i in range(self.obs_space):
            # use minimum value to anchor buckets
            start, stop = 0, ranges[i]
            buckets = np.linspace(start, stop, num_bins[i])
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

    def execute_policy(self):
        state = ''.join(str(int(elem)) for elem in self.digitize(self.env.reset()))
        #print(state, self.Policy['states'][0])
        assert(state == self.Policy['states'][0])
        
        time_step = 0
        total_rwd = 0
        terminal = False
        while not terminal:
            action = self.Policy['actions'][time_step]
            state_next, reward, terminal, info = self.env.step(action)
            total_rwd += reward; time_step += 1
        return total_rwd


    def train(self, epochs, epsilon_min=0.2, viz=False, agent=False):
        rewards = [0]; avg_rwd = 0
        EPSILON_MIN = epsilon_min
        dr_dt = 0#reward derivitive with respect to time
        
        for ep in range(1, epochs):
            epsilon = max(EPSILON_MIN, np.tanh(-ep/epochs) + 0.7)
            condition = random.random() > epsilon
            condition = condition or epsilon == EPSILON_MIN
            condition = condition and len(self.Policy['states']) > 5
            
            if condition is True:
                ep_reward = self.execute_policy()
                for i in range(len(self.Policy['actions'])):
                    s = self.Policy['states'][i]
                    a = self.Policy['actions'][i]
                    r = sum(self.Policy['rewards'])
                    self.Q[s][a] += ALPHA(r)
            else:

                zipped = self.play_episode(epsilon, viz)
            
                ep_reward, reward_arr, action_arr, state_arr = zipped
            
                if ep_reward > sum(self.Policy['rewards']):
                    self.Policy['actions'] = action_arr
                    self.Policy['rewards'] = reward_arr
                    self.Policy['states'] = state_arr

            if ep % 1 == 0:
                avg_rwd = round(np.mean(rewards),3)
                dr_dt = round(abs(dr_dt) - abs(avg_rwd), 2)
                print("Ep: {} | {}".format(ep, epochs),
                      "%:", round(ep*100/epochs, 1),
                      "Eps:", round(epsilon, 1),
                      "Avg rwd:", round(avg_rwd , 1),
                      #"Ep rwd:", int(ep_reward),
                      "dr_dt:", -dr_dt)

            if ep_reward < -1000: ep_reward = -1000
            rewards.append(ep_reward)
            dr_dt = round(avg_rwd,2)

        return rewards

    def play_episode(self, epsilon=0.2, viz=False):
        '''
        Environment must be capable of reseting and stepping
        Using self.T to explore state space
        '''
        state = self.env.reset()
        total_reward = 0
        terminal = False

        state_arr = []
        reward_arr = []
        action_arr = []
        while not terminal:
            #if viz: env.render()
            #if num_frames > 300: epsilon = 0.1

            if random.random() < epsilon:
                #action = random.randint(0, self.num_actions - 1)
                string_state = ''.join(str(int(elem)) for elem in self.digitize(state))
                action = self.sample_from_T(string_state)
            else:
                string_state = ''.join(str(int(elem)) for elem in self.digitize(state))
                action = self.get_action(string_state)
            
            state_next, reward, terminal, info = self.env.step(action)

            total_reward += reward
            reward_arr.append(reward)
            action_arr.append(action)
            state_arr.append(string_state)
            
            if terminal:
                pass#shape reward

            str_state_next = ''.join(str(int(elem)) for elem in self.digitize(state_next))
            action_next = self.get_action(str_state_next)
        
            self.update_policy(state, state_next, action, reward)
              
            state = state_next

        return total_reward, reward_arr, action_arr, state_arr
    
    def save_agent(A):
        with open('ExploreDQN.pkl', 'wb') as writer:
            pickle.dump(A, writer, protocol=pickle.HIGHEST_PROTOCOL)
        
    def load_agent(filename):
        with open(filename, 'rb') as reader:
            return pickle.load(reader)




def observe(agent, N=15):
    [play_episode(agent, EPSILON_MIN, viz=True) for ep in range(N)]

def plot_running_avg(reward_arr):
    N = len(reward_arr)
    #init unitialized array
    # (faster than np.zeros)
    running_avg = np.empty(N)

    for t in range(100, N):
        running_avg[t] = np.mean(reward_arr[t-100: t+1])

    plt.plot(running_avg, color="purple", label="Q-Learning Running Average")

def play_random(viz=False):
    observation = env.reset()
    total_reward = 0
    terminal = False

    while not terminal:
        if viz: env.render()
        action = env.action_space.sample()
        observation, reward, terminal, info = env.step(action)
        total_reward += reward
        
    return total_reward






















