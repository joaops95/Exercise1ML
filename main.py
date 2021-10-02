import numpy as np
import random
import time
from termcolor import colored
from os import system
import numpy as np
import matplotlib.pyplot as plt

class Exercise1:


    def __init__(self):
        self.board_square = colored(' ▄ ', 'white', attrs=['reverse', 'blink'])
        self.agent_icon = colored(' ㋡ ', 'red', attrs=['reverse', 'blink'])
        self.reward_icon = colored(' ❤ ', 'blue', attrs=['reverse', 'blink'])

        self.board =   [[self.board_square] * 10 for i in range(9) ] 
        self.iterations = 0
        self.reward = 0
        self.agentPosition = []
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

    def stateTransaction(self, state, action):
        self.iterations += 1
        self.agentPosition = state
        position = action()
        if(self.reachedPosition()):
            print(self.reward)
            # raise Exception
            state = self.defaultAgentPosition
        # print(position)
        return position


def plotStandarDeviation(x, y):

    # example data
    # example error bar values that vary with x-position
    error = 0.1 + 0.2 * x
    # error bar values w/ different -/+ errors
    lower_error = 0.4 * error
    upper_error = error
    asymmetric_error = [lower_error, upper_error]

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    ax0.errorbar(x, y, yerr=error, fmt='-o')
    ax0.set_title('variable, symmetric error')

    ax1.errorbar(x, y, xerr=asymmetric_error, fmt='o')
    ax1.set_title('variable, asymmetric error')
    ax1.set_yscale('log')
    plt.show()


if (__name__ == "__main__"):
    agentPosition = [0, 0]
    episodes = 5
    epochs = 1000

    rewards = []
    

    for i in range(1, episodes):
        seed = time.time()
        random.seed(seed)

        game = Exercise1()

        while(game.iterations <= epochs):
            # time.sleep(0.2)
            system('clear')

            random_action = random.randint(0, 3)
            if(random_action == 1):
                agentPosition = game.stateTransaction(agentPosition, game.moveUp)

            elif(random_action == 2):
                agentPosition = game.stateTransaction(agentPosition, game.moveDown)
            
            elif(random_action == 3):
                agentPosition = game.stateTransaction(agentPosition, game.moveRight)
            
            else:
                agentPosition = game.stateTransaction(agentPosition, game.moveLeft)

            game.print_board()
            print(f"----Reward {game.reward} Number of iteration: {game.iterations} ----")

        rewards.append(game.reward)
        
    plotStandarDeviation(np.arange(1, episodes, 1), np.array(rewards))