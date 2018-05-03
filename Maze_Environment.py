# Initialze a randomly ordered string using chars.
# The maze is partially observable but static
# --> Developed by Nathan Shepherd

class Maze():
    # An environment of Strings
    # That is initially hidden
    # Rewards and grave danger
    # --> await all who dare enter
    def __init__(self, width, depth):
        self.pos = {'x':0,'y':0}
        self.width = width
        self.depth = depth
        self.freq_map = {}
        self.counter = 0
        self.init_Map()
        
        

    def init_Map(self):
        self.Map = {}; goal = [int(self.width*0.7), int(self.depth*0.7)]
        #goal = [random.randint(int(self.width*0.8), self.width),
        #        random.randint(int(self.depth*0.8), self.depth)]
        self.goal = goal
        for row in range(self.width + 1):
            self.Map[row] = {}
            self.freq_map[row] = {}
            for col in range(self.depth + 1):
                self.freq_map[row][col] = 0
                if row == goal[0] and col == goal[1]:
                    self.Map[row][col] = 'O'
                else:
                    probs = [0.5, 0.75]
                    if random.random() <= probs[1]:
                        self.Map[row][col] = '+'
                    
                    else:
                        self.Map[row][col] = 'X'
        

    def reset(self):
        self.counter = 0
        self.pos['x'] = int(self.width*0.2)
        self.pos['y'] = int(self.depth*0.2)
        return [self.pos['x'], self.pos['y']]

    def score_pos(self):
        if self.Map[self.pos['x']][self.pos['y']] == '+':
            reward = -1
        elif self.Map[self.pos['x']][self.pos['y']] == 'X':
            reward = -2#-25
        elif self.Map[self.pos['x']][self.pos['y']] == 'O':
            reward = 500

        dist = (self.goal[0] - self.pos['x'])**2 + (self.goal[1] - self.pos['y'])**2
        dist = np.sqrt(dist)

        if not self.check_bounds: reward = -5

        if self.pos['x'] == self.goal[0] and self.goal[1] == self.pos['y']:
            reward += 100; print("GOAL!!!")

        return reward - (np.tanh(dist*100))**2# - np.tanh(self.counter)/100

    def check_if_solvable(self):
        pass# Brute force a solution if one exists

    def check_bounds(self):
        if self.pos['x'] > self.width:
            self.pos['x'] = self.width - 1; return True
        if self.pos['y'] > self.depth:
            self.pos['y'] = self.depth - 1; return True
        if self.pos['x'] < 0:
            self.pos['x'] = 1; return True
        if self.pos['y'] < 0:
            self.pos['y'] = 1; return True
        if self.pos['x'] == self.goal[0] and self.goal[1] == self.pos['y']:
            return True
        return False

    def step(self, action):
        out_of_bounds = self.check_bounds()
        info = None; self.counter += 1
        if not out_of_bounds:
            terminal = False; reward = self.score_pos()
            old_val = self.freq_map[self.pos['x']][self.pos['y']]
            self.freq_map[self.pos['x']][self.pos['y']] += 1
            
            if action == 0: self.pos['x'] += 1
            elif action == 1: self.pos['y'] -= 1
            elif action == 2: self.pos['x'] -= 1
            elif action == 3: self.pos['y'] += 1
            else: raise TypeError# Action is greater than num_actions == 4
            
        else:
            reward = -1; terminal = True

        if self.pos['x'] == self.goal[0] and self.goal[1] == self.pos['y']:
            reward = 300; terminal = True
            
        return [self.pos['x'], self.pos['y']], reward, terminal, info

    def to_string(self, binary=True):
        outs = ""
        for row in range(self.width):
            for col in range(self.depth):
                outs += self.Map[row][col]
            outs += " | "
            if binary:
                line = np.zeros(self.depth)
                for col in range(self.depth):
                    line[col] = int(self.freq_map[row][col])
                _max = max(line)
                for num in line:
                    if num == _max:
                        outs += '1'
                    else: outs += '0'
            else:
                for col in range(self.depth):
                    outs += str(int(np.tanh(self.freq_map[row][col]/EPOCHS)*10))
            outs += "\n"
        print( outs )

def observe(agent, N=15):
    [agent.play_episode(0.1, viz=True) for ep in range(N)]

def plot_running_avg(reward_arr, label, step_size=75):
    N = len(reward_arr)
    #init unitialized array
    # (faster than np.zeros)
    running_avg = np.empty(N)

    for t in range(step_size, N):
        running_avg[t] = np.mean(reward_arr[t-step_size: t+1])

    plt.plot(running_avg, color="purple", label="Running Avg.: "+label)

import matplotlib.pyplot as plt
from matplotlib import style
from StringDQN import DQN
style.use('ggplot')
import numpy as np
import random

NUM_ACTIONS = 4
WIDTH = 8# MAX: 30
DEPTH = 5# MAX: 20

EPOCHS = 500

if __name__ == "__main__":
    maze = Maze(DEPTH, WIDTH)
    initial_state = maze.reset()
    obs_space = len(initial_state)

    bin_ranges = [WIDTH, DEPTH]
    num_bins = 8#len(bin_ranges)

    Agent = DQN(obs_space, NUM_ACTIONS,
                initial_state, num_bins, bin_ranges)
    Agent.env = maze

    ep_rewards = Agent.train( EPOCHS )[1:]
    plot_running_avg(ep_rewards,'Running Avg Rwd')

    maze.to_string(binary=False)
    plt.show()
        
    print("Total Reward:", sum(ep_rewards))
