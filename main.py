from world import World, Action, NpEncoder
from qEstimationNN import DQN
import random
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
from run import getActionFromPolicy, removeStayAction



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

# def chooseAction(actionList):



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


def actionChoiceFunction(actions, stepsDone, worldDuration, epoch_num, policyNet, state):
    EPS_START = 0.9
    EPS_END = 0.1

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-10 * stepsDone / (worldDuration*epoch_num))
    
    if sample > eps_threshold:
        return getActionFromPolicy(policyNet, state, actions)
    else:
        sample = random.random()
        # Remove stay action in the beggining
        # always do so
        if len(actions) > 1:
            actions = removeStayAction(actions)
        return np.random.choice(actions)



def runOnSomeWorld(myWorld, runName):
    GAMMA = 0.99
    LR = 1e-4
    

    
    n_actions = Action(None, None,None, (0,0,0)).numPossibleActions
    n_observations = myWorld.getObservation().getObservationSpace()
    print(n_observations)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(100)
    epoch_num = 4000
    worldDuration = 300
    worldDiff = []
    losses = []
    print("Starting training")
    stepsDone = 0
    for i, epoch in enumerate(range(epoch_num)):
        print(i/epoch_num, end='\r')
        myWorld = World(8, 8, 8)
        myWorld.generateTreeWorld()
        worldDiff.append(myWorld.blocks[0,0,1])
        # np.random.seed(12)
        for i in range(worldDuration):
            stateBefore = myWorld.getObservation()
            actions = myWorld.getAgentActions()
            # action = np.random.choice(actions, )
            action = actionChoiceFunction(actions, stepsDone, worldDuration,epoch_num,policy_net,stateBefore)
            stepsDone += 1
            # print(actions)
            # print(action)
            myWorld = myWorld.executeAction(action)
            # print(myWorld.agentFuel)
            # myWorld.print()
            stateAfter = myWorld.getObservation()
            reward = myWorld.reward()
            memory.push(stateBefore.toTensor(), action.toTensor(), stateAfter.toTensor(), reward)

        batch = Transition(*zip(*memory.memory))
        state_batch = torch.stack(batch.state).float()
        action_batch = torch.stack(batch.action)
        next_state_batch = torch.stack(batch.next_state).float()
        reward_batch = torch.tensor(batch.reward)

        # reward_batch = batch.reward
        # print(state_batch.shape)
        # print(torch.cat(memory.memory[0].state.toList()))
        policyPass = policy_net(state_batch)
        state_action_values  = torch.masked_select(policyPass, action_batch == 1)
        targetPass = target_net(next_state_batch)
        next_state_values = targetPass.max(1)[0]
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        losses.append(loss.item())

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    
    if not os.path.exists(runName):
        os.makedirs(runName)

    torch.save([policy_net.kwargs, policy_net.state_dict()], runName + "/policyNetState")
    torch.save([target_net.kwargs, target_net.state_dict()], runName + "/targetNetState")


    print(np.unique(np.array(worldDiff), return_counts=True))


    N = 10
    rollingAvgLosses = np.convolve(losses, np.ones(N)/N, mode='valid')


    plt.plot(rollingAvgLosses)
    plt.show()




h = 5  # Height of the world
w = 5  # Width of the world
l = 3  # Number of layers

# myWorld = World(h, w, l)


treeWorld = World(8,8,8)
treeWorld.generateTreeWorld()

# treeWorld.print()

# targetActions = treeWorld.getHeuristicOptimalActionSeq()

# stateActionPairOut = []
# for action in targetActions:
#     print(action)
#     stateActionPairOut.append({"state": treeWorld.__dict__, 'action': action})
#     treeWorld = treeWorld.executeAction(action)
#     print(treeWorld.agentFuel)

# with open(baselineJsonPath, 'w') as file:
#     json.dump(stateActionPairOut, file, cls=NpEncoder)

runOnSomeWorld(treeWorld, "treeWorldPlantPlacing")
# actionChoiceFunction(None, 0)
# actions, stepsDone, worldDuration, epoch_num, policyNet, state
