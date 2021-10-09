import numpy as np
import random
import time
from termcolor import colored
from os import system
import numpy as np
import matplotlib.pyplot as plt
import statistics
import sys
import json
# import tkinter as tk

# from gameBoard import GameBoard
class Exercise1:


    def __init__(self, game_board, qtable):
        self.board_square = colored(' ▄ ', 'white', attrs=['reverse', 'blink'])
        self.agent_icon = colored(' ▄ ', 'red', attrs=['reverse', 'blink'])
        self.reward_icon = colored(' ▄ ', 'blue', attrs=['reverse', 'blink'])
        self.board = game_board 
        self.iterations = 0
        self.game_iterations = 1

        self.reward = 0
        self.alpha = 0.1
        self.discount = 0.5
        self.agentPosition = [0, 0]
        self.qtable = qtable
        self.defaultAgentPosition = [0, 0]
        # print(len(self.board[1]))
        self.goalPosition = [len(self.board) -1 , len(self.board[0]) -1]

    # for printing the board
    def print_board(self):
        print(self.goalPosition[0],self.goalPosition[1])

        for ix, i in enumerate(self.board):
            for jx, j in enumerate(i):
                if(self.agentPosition == self.goalPosition):
                    if(ix == self.agentPosition[0] and jx == self.agentPosition[1]): print(self.agent_icon, end = '')
                    else: print(j, end = '')                
                else:
                    if(ix == self.agentPosition[0] and jx == self.agentPosition[1]): print(self.agent_icon, end = '')
                    else: print(j, end = '')
                    if(ix == self.goalPosition[0] and jx == self.goalPosition[1]): print(self.reward_icon, end = '')
                    else: print(j, end = '')
            print('|')


    def moveUp(self):
        if(self.agentPosition[0]-1 >= 0):
            self.agentPosition = [self.agentPosition[0]-1, self.agentPosition[1]]
        return [self.agentPosition[0], self.agentPosition[1]]
    def moveDown(self):
        if(self.agentPosition[0]+1 < len(self.board)):
            self.agentPosition = [self.agentPosition[0]+1, self.agentPosition[1]]
        return [self.agentPosition[0], self.agentPosition[1]]

    def moveRight(self):

        if(self.agentPosition[1]+1 < len(self.board[0])):
            self.agentPosition = [self.agentPosition[0], self.agentPosition[1]+1]
        return [self.agentPosition[0], self.agentPosition[1]]

    def moveLeft(self):
        if(self.agentPosition[1]-1 >= 0):
            self.agentPosition = [self.agentPosition[0], self.agentPosition[1]-1]
        return [self.agentPosition[0], self.agentPosition[1]]

    def reachedPosition(self):

        if(self.agentPosition == self.goalPosition):
            self.reward += 100
            self.game_iterations = 1
            return True
        self.reward += 0
        return False

    def mapActionNameToIndex(self, action_name):
        res = isinstance(action_name, str)
        if(res):
            if(action_name == 'moveUp'): return 0
            if(action_name == 'moveDown'): return 1
            if(action_name == 'moveRight'): return 2
            if(action_name == 'moveLeft'): return 3
        else:
            if(action_name == 0): return self.moveUp
            if(action_name == 1): return self.moveDown
            if(action_name == 2): return self.moveRight
            if(action_name == 3): return self.moveLeft

    def getAgentStatePosition(self, agentPosition):
     
        return agentPosition[0]*np.asarray(self.board).shape[1]+agentPosition[1]

    def updateQValues(self, state_index,action_index):
        reward_s_a = self.reward

        q_s_a = self.qtable[state_index][action_index]
        if(state_index + 1 == len(np.asarray(self.board).flatten())): next_state_index = state_index - 1
        else: next_state_index = state_index+1
        q_ss_aa = self.qtable[next_state_index]
        alpha = self.alpha
        self.qtable[state_index][action_index] = (1-alpha)*q_s_a+alpha*(reward_s_a + self.discount * np.max(q_ss_aa))
        np.savetxt('text.txt',self.qtable,fmt='%.2f')

    def stateTransaction(self, state, action):

        self.updateQValues(self.getAgentStatePosition(self.agentPosition), self.mapActionNameToIndex(action.__name__))
        self.iterations += 1
        self.game_iterations += 1
        self.agentPosition = state
        position = action()
        if(self.reachedPosition()):
            # raise Exception
            state = self.defaultAgentPosition
        # print(position)
        return position


    def runTest(self, iterations):
        game_test = Exercise1(self.board, self.qtable)

        it = 0
        agentPosition = [0, 0]
        rewards = []
        while(it < iterations):
            it += 1
            action_space_index = game_test.getAgentStatePosition(agentPosition)
            action_space = self.qtable[action_space_index]
            # raise Exception
            random_action = np.unravel_index(np.argmax(action_space, axis=None), action_space.shape)[0]
            print("RANDOM MACTIONS")
            print(random_action)
            system('clear')
            if(random_action == 0):
                agentPosition = game_test.stateTransaction(agentPosition, game_test.moveUp)

            elif(random_action == 1):
                agentPosition = game_test.stateTransaction(agentPosition, game_test.moveDown)
            
            elif(random_action == 2):
                agentPosition = game_test.stateTransaction(agentPosition, game_test.moveRight)
            
            else:
                agentPosition = game_test.stateTransaction(agentPosition, game_test.moveLeft)

            game_test.print_board()
            
            print(f"----Reward {game_test.reward} Number of iteration: {game_test.iterations} ----")
        
        return game_test



def menu():
    print("************Welcome to Lab 1 AI**************")

    choice = input("""
                      A: Run env with Qtable from random actions
                      B: Run env with Qtable from self values
                      C: Run env with increasing E-greedy random
                      D: Run env with increasing E-greedy random, include episodes

                      Please enter your choice: """)

    if choice == "A" or choice =="a":
        runEnv(1)
    elif choice == "B" or choice =="b":
        runEnv(0)
    elif choice == "C" or choice =="c":
        runEnv(1, True)
    elif choice == "D" or choice =="d":
        runEnv(1, True, True)
    elif choice=="Q" or choice=="q":
        sys.exit
    else:
        print("You must only select either A or B")
        print("Please try again")
        menu()




def runEnv(random_probability = 1, auto_increase = False, include_episodes = True):
    numberOfTests = 1
    test_iterations = 1000
    trainResults = []
    testResults = {}
    num_cols = 10
    num_rows = 10
    random_probability = random_probability
    qtable_probability = 1 - random_probability

    testEpochs = [100, 200, 500, 600, 700, 800, 900, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 25000, 30000]
    game_board = [[colored(' ▄ ', 'white', attrs=['reverse', 'blink'])] * num_cols for i in range(num_rows) ]
    
        
    
    for j in range(1, numberOfTests+1):
        agentPosition = [0, 0]
        episodes = 2
        epochs = 31000
        rewards = []
        qtable = np.zeros(shape=(len(np.asarray(game_board).flatten()), 4))
        curr_len = 0

        for episode in range(1, episodes+1):
            curr_factor = episode
            if(not include_episodes): 
                curr_len = 0
                curr_factor = 1
            seed = time.time()
            random.seed(seed)

            game = Exercise1(game_board, qtable)

            while(game.iterations < epochs - 1):
                # time.sleep(0.2)
                system('clear')
                if(game.iterations in testEpochs):
                    curr_len += 1
                    if(auto_increase): 
                        # [1 , 0]
                        qtable_probability = curr_len/ (len(testEpochs)*curr_factor)
                        random_probability = 1 - curr_len/(len(testEpochs)*curr_factor)

                    print("random_probability")
                    print(qtable_probability)
                    game_test = game.runTest(test_iterations)
                    # stdev = statistics.stdev(rewards)
                    try:
                        testResults[episode]['results'].append({
                            'epoch':game.iterations,
                            'reward':game_test.reward/test_iterations
                        })
                    except KeyError:
                        testResults[episode] = {}
                        testResults[episode]['results'] = [
                            {
                            'epoch':game.iterations,
                            'reward':game_test.reward/test_iterations
                            }
                        ]

                    
                    with open('results.json', 'w') as outfile:
                        json.dump(testResults, outfile, indent=4, sort_keys=True)

                    system('clear')

                a_list = [1, 2]

                distribution = [random_probability, qtable_probability]
                

                random_number = random.choices(a_list, distribution)
                
                print("random_probability")
                print(distribution)
                print(random_number[0])

                if(random_number[0] == 1):
                    random_action = random.randint(0, 3)
                    if(random_action == 0):
                        
                        agentPosition = game.stateTransaction(agentPosition, game.moveUp)
                    elif(random_action == 1):
                        agentPosition = game.stateTransaction(agentPosition, game.moveDown)
                    elif(random_action == 2):
                        agentPosition = game.stateTransaction(agentPosition, game.moveRight)
                    else:
                        agentPosition = game.stateTransaction(agentPosition, game.moveLeft)
                else:

                    action_space_index = game.getAgentStatePosition(agentPosition)
                    print(action_space_index)
                    action_space = game.qtable[action_space_index]
                    # raise Exception
                    random_action = np.unravel_index(np.argmax(action_space, axis=None), action_space.shape)[0]
                    if(random_action == 0):
                        
                        agentPosition = game.stateTransaction(agentPosition, game.moveUp)
                    elif(random_action == 1):
                        agentPosition = game.stateTransaction(agentPosition, game.moveDown)
                    elif(random_action == 2):
                        agentPosition = game.stateTransaction(agentPosition, game.moveRight)
                    else:
                        agentPosition = game.stateTransaction(agentPosition, game.moveLeft)


            heatmap = game.qtable.mean(1)/(game.iterations*episode)
            heatmap = heatmap.reshape(num_rows, num_cols)
            # a = np.random.random((16, 16))
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            plt.savefig(f'assets/heatmap_ep{episode}.png')
            testResults[episode]['heatmap_fig'] = f'assets/heatmap_ep{episode}.png'


            # game.print_board()
            # print(f"----Reward {game.reward} Number of iteration: {game.iterations} ----")

            rewards.append(game.reward)
            qtable = game.qtable
        
        print(rewards)
        stdev = statistics.stdev(rewards)


        trainResults.append({
            'seed':seed,
            'rewards':rewards,
            'mean': np.mean(rewards),
            'min': np.min(rewards),
            'max':np.max(rewards),
            'stdev': stdev
        })

if (__name__ == "__main__"):

    menu()


    # for test in trainResults:
    #     # example data
    #     # example error bar values that vary with x-position
    #     x = np.arange(1, episodes, 1)
    #     rewards_y = np.asarray(test['rewards'])
    #     y = np.array(rewards_y)
    #     error = test['stdev']*np.arange(1, episodes, 1)
    #     # error bar values w/ different -/+ errors
        
    #     lower_error = abs((test['mean'])/test['min'])*rewards_y
    #     upper_error = abs((test['mean'])/test['max'])*rewards_y
    #     asymmetric_error = [lower_error, upper_error]
        
    #     fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    #     ax0.errorbar(x, y, yerr=error, fmt='-o')
    #     ax0.set_title(f'variable, symmetric error seed {test["seed"]}')

    #     ax1.errorbar(x, y, xerr=asymmetric_error, fmt='o')
    #     ax1.set_title(f'variable, asymmetric error seed {test["seed"]}')
    #     ax1.set_yscale('log')
    #     plt.show()