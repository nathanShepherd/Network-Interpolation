# Find an efficent route to the goal in a Maze
# Using Model-Free Machine learning
# Developed by Nathan Shepherd

def observe(agent, N=15):
    [agent.play_episode(0.1, viz=True) for ep in range(N)]

def plot_running_avg(reward_arr, label, step_size=75):
    N = len(reward_arr)
    #init unitialized array
    # (faster than np.zeros)
    running_avg = []

    for t in range(step_size, N):
        mu = np.mean(reward_arr[t-step_size: t+1])
        running_avg.append( mu )

    plt.plot(running_avg, color="purple", label=label)

def use_DQN(maze, which='String'):
    initial_state = maze.reset()
    obs_space = len(initial_state)

    bin_ranges = [[WIDTH, WIDTH], [DEPTH, DEPTH]]
    num_bins = max(WIDTH, DEPTH)
    
    if which == 'String':
        Agent = StringDQN(obs_space, NUM_ACTIONS,
                initial_state, num_bins, bin_ranges)
    if which == 'StateGreedy':
        Agent = SGDQN(obs_space, NUM_ACTIONS,
                      initial_state, bin_ranges)
    Agent.env = maze

    ep_rewards = Agent.train( EPOCHS, epsilon_min=0.01 )[1:]
    plt.title(which + "DQN Performance")
    plt.ylabel("Reward")
    plt.xlabel("Epochs")
    plot_running_avg(ep_rewards,'Running Avg Rwd')

    maze.to_string(binary=False)
    plt.legend()
    plt.show()
        
    print("Total Reward:", sum(ep_rewards))
    return Agent

from StateGreedyDQN import DQN as SGDQN
from StringDQN import DQN as StringDQN
from Maze_Environment import Maze
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import random

NUM_ACTIONS = 4
WIDTH = 75# MAX: 40
DEPTH = 10# MAX: 30

EPOCHS = 10**5

'''
StringDQN stabilized at -20 after 1000 iterations on a board of size [WIDTH=20, DEPTH=10]

      Maze to solve | Frequency of Agent in State   Max action by row&col
  00123456677888899999 @ 00123456677888899999 @ 00123456677888899999
  ------------------------------------------------------------------
0 +X+++X++X++X+++++X++ | 01112242310000000000 | 00000010000000000000
0 ++XX++++X+X+X+X++++X | 02346764421000000000 | 00000100000000000000
0 +X++++X+++++++X++X++ | 13479964321000000000 | 00001000000000000000
1 +XXX+++X++X+X+++++++ | 12469874331000000000 | 00001000000000000000
1 X+X++++X++++++++XX++ | 12578763321100000000 | 00001000000000000000
2 XX+XX++++++X++X++X+X | 12455673321100000000 | 00000010000000000000
2 X++++++XX+++++++X++X | 12455563221100000000 | 00000010000000000000
3 ++++X++X++X++XO++X+X | 13444463221100000000 | 00000010000000000000
3 +++++X++++++++++++++ | 13344353331000000000 | 00000010000000000000
4 X++X+X+++++X++X+++++ | 02333343321000000000 | 00000010000000000000

StateGreedyDQN stabilized at 200 after <1000 iterations on a board of size [WIDTH=20, DEPTH=10]
  00011223344555666677 @ 00011223344555666677 @ 00011223344555666677
  ------------------------------------------------------------------
0 X+X++XXX+XX++++X++XX | 11131899911113111111 | 00000000100000000000
0 ++++X++++++X+++++++X | 11453999999115111111 | 00000001000000000000
1 +X++X++++X+X+++X++X+ | 11459999989899311111 | 00000001000000000000
2 ++++++++XXX++++X+++X | 11234335213399991111 | 00000000000010000000
3 +X++X++X+X++++++++++ | 11222222211111191111 | 00000000000000010000
4 +++X+X+++X++X+++++++ | 11212121111111191111 | 00000000000000010000
5 +X+++XXXX++X+X+++XXX | 11111111111111991111 | 00000000000000110000
6 +++++++++X++X+O+++X+ | 11111111111111111111 | 00010000000000000000
6 XX++XX+X++X++X++++X+ | 11111111111111111111 | 00010000000000000000
7 ++++++++++X++X+X+X++ | 11111111111111111111 | 00010010000000000000


On board of size [WIDTH=30, DEPTH=20]
    StateGreedy stabilized a Policy with 200 avg reward after 5000 Episodes
    StringDQN Failed to converge
'''

if __name__ == "__main__":
    maze = Maze(DEPTH, WIDTH)
    Agent = use_DQN(maze, "StateGreedy")
    

    









    
