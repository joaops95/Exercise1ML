import numpy as np
import random
import time
from termcolor import colored
from os import system
import numpy as np
import matplotlib.pyplot as plt
import statistics

class Exercise1:


    def __init__(self, game_board, qtable):
        self.board_square = colored(' ▄ ', 'white', attrs=['reverse', 'blink'])
        self.agent_icon = colored(' ▄ ', 'red', attrs=['reverse', 'blink'])
        self.reward_icon = colored(' ▄ ', 'blue', attrs=['reverse', 'blink'])

        self.board =   game_board 
        self.iterations = 0
        self.reward = 0
        self.alpha = 0.7
        self.discount = 0.99
        self.agentPosition = []
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
        reward_s_a = self.reward/self.iterations
        q_s_a = self.qtable[state_index - 1][action_index]
        q_ss_aa = self.qtable[state_index]
        alpha = self.alpha
        self.qtable[state_index][action_index - 1] = (1-alpha)*q_s_a+alpha*(reward_s_a + self.discount * np.max(q_ss_aa))
        print(self.qtable[state_index][action_index - 1])
        np.savetxt('text.txt',self.qtable,fmt='%.2f')

    def stateTransaction(self, state, action):
        self.iterations += 1
        self.agentPosition = state

        print(self.iterations)
        print(self.getAgentStatePosition(self.agentPosition))
        self.updateQValues(self.getAgentStatePosition(self.agentPosition), self.mapActionNameToIndex(action.__name__))
        position = action()
        if(self.reachedPosition()):
            print(self.reward)
            # raise Exception
            state = self.defaultAgentPosition
        # print(position)
        return position


    def runTest(self):
        game_test = Exercise1(game_board, qtable)

        it = 0
        agentPosition = [0, 0]

        while(it < 999):
            action_space = game_test.getAgentStatePosition(agentPosition)
            print(action_space)
            ind = np.unravel_index(np.argmax(action_space, axis=None), action_space.shape)

            print(ind)
            if(random_action == 1):
                agentPosition = game_test.stateTransaction(agentPosition, game_test.moveUp)

            elif(random_action == 2):
                agentPosition = game_test.stateTransaction(agentPosition, game_test.moveDown)
            
            elif(random_action == 3):
                agentPosition = game_test.stateTransaction(agentPosition, game_test.moveRight)
            
            else:
                agentPosition = game_test.stateTransaction(agentPosition, game_test.moveLeft)


if (__name__ == "__main__"):
    numberOfTests = 2
    testResults = []
    game_board = [[colored(' ▄ ', 'white', attrs=['reverse', 'blink'])] * 10 for i in range(10) ]
    for j in range(1, numberOfTests):
        agentPosition = [0, 0]
        episodes = 5
        epochs = 20000
        rewards = []
        print(game_board)
        # print()
        # raise Exception
        qtable = np.zeros(shape=(len(np.asarray(game_board).flatten()), 4))

        for i in range(1, episodes):
            seed = time.time()
            random.seed(seed)
            # 4 porque temos 4 acoes possiveis?
            print(qtable.shape)
            # raise Exception
            game = Exercise1(game_board, qtable)

            while(game.iterations < epochs - 1):
                # time.sleep(0.2)
                # system('clear')
                if(game.iterations == 1000):
                    game.runTest()
                random_action = random.randint(0, 3)
                if(random_action == 1):
                    agentPosition = game.stateTransaction(agentPosition, game.moveUp)

                elif(random_action == 2):
                    agentPosition = game.stateTransaction(agentPosition, game.moveDown)
                
                elif(random_action == 3):
                    agentPosition = game.stateTransaction(agentPosition, game.moveRight)
                
                else:
                    agentPosition = game.stateTransaction(agentPosition, game.moveLeft)

                # game.print_board()
                print(f"----Reward {game.reward} Number of iteration: {game.iterations} ----")

            rewards.append(game.reward)
            qtable = game.qtable
        stdev = statistics.stdev(rewards)


        testResults.append({
            'seed':seed,
            'rewards':rewards,
            'mean': np.mean(rewards),
            'min': np.min(rewards),
            'max':np.max(rewards),
            'stdev': stdev
        })



    for test in testResults:
        # example data
        print(test)
        # example error bar values that vary with x-position
        x = np.arange(1, episodes, 1)
        rewards_y = np.asarray(test['rewards'])
        y = np.array(rewards_y)
        error = test['stdev']*np.arange(1, episodes, 1)
        # error bar values w/ different -/+ errors
        
        lower_error = abs((test['mean'])/test['min'])*rewards_y
        upper_error = abs((test['mean'])/test['max'])*rewards_y
        print(lower_error)
        print(upper_error)
        asymmetric_error = [lower_error, upper_error]
        
        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
        ax0.errorbar(x, y, yerr=error, fmt='-o')
        ax0.set_title(f'variable, symmetric error seed {test["seed"]}')

        ax1.errorbar(x, y, xerr=asymmetric_error, fmt='o')
        ax1.set_title(f'variable, asymmetric error seed {test["seed"]}')
        ax1.set_yscale('log')
        plt.show()