from world import World, Action
from qEstimationNN import DQN
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from copy import deepcopy



from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory():


    def __init__(self, capacity):
        self.maxReward = 0
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        transitions = Transition(*args)
        self.maxReward = max(transitions.reward, self.maxReward)
        self.memory.append(transitions)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

def getActionFromPolicy(policyNet, state, actions):
    oneHotActions = [action.toTensor() for action in actions]
    
    policyPass = policyNet(state.toTensor().float())

    bestAction = actions[0]
    qMax = 0
    for i, action in enumerate(oneHotActions):
        qValue = policyPass[np.argmax(action)]
        if qMax < qValue:
            qMax = qValue
            bestAction = actions[i]
    return bestAction


def removeStayAction(actionList):
    actions = deepcopy(actionList)
    for action in actions:
        if action.actionType == 'move' and action.posChange == (0,0,0):
            actions.remove(action)
    return actions


def runWithTrainedPolicy(runDirectory, myWorld, outputStateActionPairPath):

    GAMMA = 0.99
    LR = 1e-4

    targetNetState = torch.load(runDirectory + '/targetNetState')
    n_actions = targetNetState['layer3.bias'].shape[0]
    n_observations = myWorld.getObservation().getObservationSpace()
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    # for el in torch.load(runDirectory + '/targetNetState').values():
    #     print(el.shape)
    target_net.load_state_dict(targetNetState)
    target_net.eval()

    policy_net.load_state_dict(torch.load(runDirectory + '/policyNetState'))
    policy_net.eval()


    memory = ReplayMemory(100)
    memoryNonTensor = ReplayMemory(100)
    epoch_num = 2000
    worldDiff = []
    losses = []
    worldDiff.append(myWorld.blocks[0,0,1])
    # np.random.seed(12)

  
    rewards = []
    actionsTaken = []
    observations = []
    states =[]
    for i in range(200):
        stateBefore = myWorld.getObservation()
        states.append(myWorld.__dict__)
        observations.append(stateBefore)
        actions = myWorld.getAgentActions()

        # assert(actions[-1].posChange == (0,0,0) and actions[-1].p)
        if len(actions) > 1 and random.uniform(0, 1) < 0.8:
            actions = removeStayAction(actions)

            
        # print(stateBefore)
        # print(actions)
        # oneHotActions = [action.toTensor() for action in actions]

        
        # policyPass = policy_net(stateBefore.toTensor().float())

        # bestAction = actions[0]
        # qMax = 0
        # for i, action in enumerate(oneHotActions):
        #     qValue = policyPass[np.argmax(action)]
        #     if qMax < qValue:
        #         qMax = qValue
        #         bestAction = actions[i]


        bestAction = getActionFromPolicy(policy_net, stateBefore, actions)
        # print(bestAction)

        myWorld = myWorld.executeAction(bestAction)
        actionsTaken.append(bestAction)
        # print(actionsTaken[0])
        # print(actionsTaken[0].__dict__)
        # break
        rewards.append(myWorld.reward())


    # myWorld.print()
    print(rewards)
    # print(states)
    stateActionPairs = [{'state': s, 'action': a} for s, a in zip(states,actionsTaken)]
    with open(outputStateActionPairPath, 'w') as file:
        json.dump(stateActionPairs, file, cls=NpEncoder)
    # # for s, a in zip(observations, actionsTaken):
    #     print(s,a)


    

h = 5  # Height of the world
w = 5  # Width of the world
l = 3  # Number of layers

# myWorld = World(h, w, l)
# myWorld.generate()


treeWorld =  World(8, 8, 8)
treeWorld.generateTreeWorld()

treeRegrowPath = '.\\turtleRunStateActionsTreeRegrow.json'
runDirectory = "treeWorldPlantPlacing"
runWithTrainedPolicy(runDirectory, treeWorld, treeRegrowPath)


