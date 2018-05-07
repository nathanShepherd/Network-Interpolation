# Initialze a randomly ordered string using chars.
# The maze is partially observable but static
#
# state( O ) is the goal and worth 500 pts
# state( + ) is a normal tile and worth -1 pts
# state( X ) is to be avoided, worth -25 pts
# Agent is rewarded for being closer to the goal
#
# --> Developed by Nathan Shepherd

import numpy as np
import random

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
            reward = -10
        elif self.Map[self.pos['x']][self.pos['y']] == 'O':
            reward = 500

        dist = (self.goal[0] - self.pos['x'])**2 + (self.goal[1] - self.pos['y'])**2
        dist = np.sqrt(dist)

        if not self.check_bounds: reward = -5

        if self.pos['x'] == self.goal[0] and self.goal[1] == self.pos['y']:
            reward += 100; print("GOAL!!!")

        return reward - (np.tanh(dist*100))*2# - np.tanh(self.counter)/100

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
            reward = -100; terminal = True

        if self.pos['x'] == self.goal[0] and self.goal[1] == self.pos['y']:
            reward = 300; terminal = True
            
        return [self.pos['x'], self.pos['y']], reward, terminal, info

    def to_string(self, config=['maze','freq'], binary=False):
        outs = "  "
        for col in range(self.depth):# make column labels
            if 'freq' in config:
                outs += str(int(np.tanh(col/self.depth)*10))
        outs += " @ "
        for col in range(self.depth):# make column labels
            if 'maze' in config:
                outs += str(int(np.tanh(col/self.depth)*10))
        outs += '\n'
        
        for row in range(self.width):# make row labels
            outs += str(int(np.tanh(row/self.width)*10)) + " "
            for col in range(self.depth):
                if 'maze' in config:
            
                    if row == int(self.width*0.2) and col == int(self.depth*0.2):
                        outs += '&'
                    else:
                        outs += self.Map[row][col]

            outs += " | "
            
            if binary and 'freq' in config:
                # TODO: Sample from Q and select max actions
                line = np.zeros(self.depth)
                for col in range(self.depth):
                    line[col] = int(self.freq_map[row][col])

                col_max = max(line)
                row_max = max(self.freq_map[row])
                    
                for num in line:
                    if num == row_max or num == col_max:
                        outs += '1'
                    else: outs += '0'
            elif 'freq' in config:
                for col in range(self.depth):
                    _max = max(self.freq_map[row])*20
                    col = min(9, int(np.tanh(self.freq_map[row][col]/_max)*10))
                    if self.freq_map[row][col] > 0 and col == 0: col = 1
                    outs += str(col)
            outs += "\n"
        print( outs )

