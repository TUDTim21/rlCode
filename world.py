import numpy as np
import math
import itertools
from copy import deepcopy
from enum import Enum
import random
import torch
import json
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, TreeGrowTime):
            return obj.__dict__
        if isinstance(obj, Action):
            return removePrivateFields(obj.__dict__)
            # return {'type':, obj[0], }
        return super(NpEncoder, self).default(obj)


def removePrivateFields(input_dict):
    return {key: value for key, value in input_dict.items() if not key.startswith('_')}


class Action:
    def __init__(self, actionType, newPosition, cost:int, posChange:tuple, placeDirection:tuple = None, itemToPlace=None):
        self.actionType = actionType
        self.newPosition = newPosition
        self.posChange = tuple(posChange)
        self.cost = cost
        # self.actionSpace = getActionSpace()
        possiblePosChanges = []
        for step in [-1, 1]:
            possiblePosChanges.append((step, 0, 0))
            possiblePosChanges.append((0, step, 0))
            possiblePosChanges.append((0, 0, step))
        actions = []
        for i, moveType in enumerate(['move', 'break', 'place']):
            for posChange in possiblePosChanges:
                actions.append((moveType, posChange))
        actions.append(('move', (0,0,0)))
        actions.append(('refuel', (0,0,0)))
        self._possibleActions = actions
        self.numPossibleActions = len(self._possibleActions)
        self._actionTypesPossible = ['move', 'break', 'refuel', 'place']
        self._numActionTypesPossible = len(self._actionTypesPossible)
        self.placeDirection = placeDirection
        self.itemToPlace = itemToPlace

    def __str__(self):
        outString = f"({self.actionType},\t newPos: {self.newPosition}, \tcost: {self.cost}, "
        if self.actionType == 'place':
            outString += f"\tplce: {self.placeDirection})"
        else:
            outString += f"\tchng: {self.posChange})"
        return outString

    def __repr__(self):
        return str(self)



    def toTensor(self):
        # onehotType = np.zeros(self._numActionTypesPossible, dtype=int)
        # onehotType[self._actionTypesPossible.index(self.actionType)] = 1
        
        # x, y, z = self.posChange
        # oneHotMove = np.zeros(self._numPossiblePosChange, dtype=int)
        # print(self.posChange, x, y, z)
        
        # onehotMoveInd = self._possiblePosChanges.index(self.posChange)
        

        # oneHotMove[onehotMoveInd] = 1 
        # return torch.tensor(np.append(onehotType, oneHotMove))
        oneHotEncoded = np.zeros(self.numPossibleActions, dtype=int)
        actionInfo = (self.actionType, self.posChange)
        if self.actionType == 'place':
            actionInfo = (self.actionType, self.placeDirection)
        index = self._possibleActions.index(actionInfo)
        oneHotEncoded[index] = 1
        return torch.tensor(oneHotEncoded)


        
        # for change in self._possiblePosChanges:
        #     test = np.zeros(self._numPossiblePosChange, dtype=int)
        #     print(change)
        #     onehotMoveInd = self._possiblePosChanges.index(change)
        #     test[onehotMoveInd] = 1 
        #     print(test)
        

        # Incase optimization is needed
        # dirFunc = lambda x: int(math.floor(x + 1 + abs(x) - 1) + abs(x-1)/2)
        # dimFunc = lambda x: int(math.ceil(dirFunc(x)/2))
        # adjFunc = lambda x: abs(x-1)
        # for x, y, z in possiblePosChanges:
        #     print(x,y,z)
        #     print(dirFunc(x), dirFunc(y), dirFunc(z))
        #     print(dimFunc(x), dimFunc(y), dimFunc(z))
        #     # print([dimFunc(x),dimFunc(y),dimFunc(z)].index(1))
        #     # onehotMoveInd =  dirFunc(x=x) + dirFunc(x=y) + 1 * dimFunc(x=y) + dirFunc(x=z) + 2 * dimFunc(x=z)
        #     onehotMoveInd = abs(dirFunc(x=x)-1) + abs(dirFunc(x=y)-1) + abs(dirFunc(x=z)-1) + 2 * dimFunc(x=y) + 4 * dimFunc(x=z)
        #     print("dimfunc", onehotMoveInd)

        # for i, cell in enumerate(self.cellContents):
        #     oneHotEncodedCells[i * self._numObservalBlocks + cell - self._observableBlocks[0]] = 1
        # encodedCells = torch.tensor(oneHotEncodedCells)
        # return torch.cat((encodedCells, torch.tensor([self.agentStorage, self.agentFuel])))


BLOCK_IDS = range(-2, 5)
BLOCKS_DICT = {'wood': 2, 'sapling': 3, 'dirt': 1, 'leaves':4, 'air': 0}


class Observation:

    def __init__(self, cellContents, agentStorage, agentFuel):
        self._maxStorage = 16
        self._maxFuel = 32
        assert(len(cellContents) == 6)
        self.cellContents = cellContents
        self.agentStorage = agentStorage[2]
        self.agentFuel = agentFuel
        self._observableBlocks = sorted(BLOCK_IDS)
        self._numObservalBlocks = len(self._observableBlocks)

    def __str__(self):
        return f"Obs(adjCells:{self.cellContents},\t agStor:{self.agentStorage},\t fuel:{self.agentFuel})"

    def getObservationSpace(self):
        
        # adjCells:[-2, -2, 1, 0, 2, 0],       agStor:1,       fuel:0)
        # Fuel and rewardBlock indicator + 6 * numberOfObservableBlocks in world [onehotencoded]
        return 2 + len(self._observableBlocks * 6) 

    def __repr__(self):
        return str(self)
    
    def toTensor(self): 
        # cellContents = np.transpose(np.array(self.cellContents)).reshape(-1,1)
        oneHotEncodedCells = np.zeros(self._numObservalBlocks * 6, dtype=int)
        for i, cell in enumerate(self.cellContents):
            oneHotEncodedCells[i * self._numObservalBlocks + cell - self._observableBlocks[0]] = 1
        encodedCells = torch.tensor(oneHotEncodedCells)
        # for 
        # cellContents = torch.tensor(self.cellContents)
        # print(np.array(self._observableBlocks).reshape(-1,1))
        # o_encoder = OneHotEncoder(categories=np.array(self._observableBlocks).reshape(-1,1)..)
        # print(o_encoder.fit_transform(cellContents))
        
        # print(cellContents)
        # cellContents = F.one_hot(cellContents, num_classes=self._numObservalBlocks)
        # print(cellContents)
        return torch.cat((encodedCells, torch.tensor([self.agentStorage, self.agentFuel])))


class SearchNode:
    def __init__(self, state, action=None):
        self.state = state
        self.action = action
        self.children = []

    def add_child(self, child):
        self.children.append(child)


class Turtle:
    def __init__ (self, agentId, position):
        self.agentId = agentId
        self.agentStorage = [0] * 16
        self.agentFuel = 0
        self.agentPos = position

class TreeGrowTime:
    def __init__(self, position: tuple, growthDoneTick:int):
        self.position = position
        self.growthDoneTick = growthDoneTick
    
    def __str__(self):
        return f"({self.position}, {self.growthDoneTick})"
    
    def __repr__(self):
        return str(self)

class World:
    def __init__(self, h, w, l):
        self.w = w
        self.h = h
        self.l = l
        self.dimensions = (w, h, l)
        # FIXME: should be w, l, h
        # width x, layers y, height z
        self.blocks = np.zeros((w, h, l)).astype(int)
        self.agentId = -1
        self.agentStorage = [0] * (max(BLOCKS_DICT.values()) + 1)
        self.agentStorage[2] = 2
        self.agentFuel = 1
        self.worldTick = 0
        self.treeToGrow = []

        # self.generate()
    def resetWorld(self):
        return self._initalWorld

    
    def generate(self):
        self.blocks[:, :, 0] = np.ones((self.w, self.h))
        for layer in range(1, self.l):
            self.blocks[:, :, layer] = np.random.choice([0, 2], size=(self.w, self.h), p=[0.8, 0.2])
        self.agentPos = (np.where(self.blocks==0)[0][0], np.where(self.blocks==0)[1][0], np.where(self.blocks==0)[2][0])
        self.blocks[self.agentPos] = self.agentId

    
    def generateTreeWorld(self):
        self.blocks[:, :, 0] = np.ones((self.w, self.h))
        treeTrunks = []

        padding = 2
        treePossible = np.ones((self.w + 2* padding, self.h + 2*padding))
        for edge in range(-3, 3):
            treePossible[edge,:] = 0
            treePossible[:,edge] = 0

        posx, posy = np.where(treePossible >= 0.1)
        remainingPositions = np.transpose(np.array((posx, posy)))

        treeKernel = np.array([[0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0, 0, 0, 0.5], [0.5, 0, 0, 0, 0.5], [0.5, 0, 0, 0, 0.5],[0.5, 0.5, 0.5, 0.5, 0.5]])

        while remainingPositions.any():
            choice = random.randrange(0, len(remainingPositions))
            x, y = remainingPositions[choice,:]
            if treePossible[x,y] < np.random.uniform():
                treePossible[x,y] = 0

            treeTrunks.append((x-padding,y-padding))
            treePossible[x-padding:x+padding+1, y-padding: y+padding+1] = np.multiply(treePossible[x-padding:x+padding+1, y-padding: y+padding+1], treeKernel)
            posx, posy = np.where(treePossible >= 0.1)
            remainingPositions = np.transpose(np.array((posx, posy)))
            # remainingPositions = np.delete(remainingPositions, choice,0)

            # for val in [x,y]:
                # for delProb in symmetricKernel:
        for trunk in treeTrunks:
            height = random.randrange(3,6)
            for leafX in range(trunk[0]-1, trunk[0] + 2):
                for leafY in range(trunk[1]-1, trunk[1] + 2):
                    self.blocks[leafX, leafY,height] = BLOCKS_DICT['leaves']
            self.blocks[trunk[0],trunk[1],1+height] = BLOCKS_DICT['leaves']
            self.blocks[trunk[0],trunk[1],1:1+height] = BLOCKS_DICT['wood']
        
        # Ground level 1, dirt is on 0
        spawnSpots = np.where(self.blocks[:,:,1]==0)
        midIndex = spawnSpots[0].size//2
        self.agentPos = (spawnSpots[0][midIndex], spawnSpots[1][midIndex], 1)
        # self.agentPos = (np.where(self.blocks==0)[0][0], np.where(self.blocks==0)[1][0], np.where(self.blocks==0)[2][0])
        self.blocks[self.agentPos] = self.agentId

    def plantTree(self, position):
        ''' At certain location grow tree (TOFIX: if space above is available)
        '''
        blockBelow = self.blocks[tuple(np.subtract(position, (0,0,1)))]
        assert(blockBelow == BLOCKS_DICT['dirt'])
        height = random.randrange(3,6)
        assert(height + position[2] <= self.l)
        for leafX in range(max(0,position[0]-1), min(self.w, position[0] + 2)):
            for leafY in range(max(0, position[1]-1), min(self.h, position[1] + 2)):
                leafPos = (leafX, leafY,min(self.l, position[2] + height))

                if self.blocks[leafPos] == BLOCKS_DICT['air']:
                    self.blocks[leafPos] = BLOCKS_DICT['leaves']
        
        if self.blocks[position[0],position[1],position[2] + 1+height] == BLOCKS_DICT['air']:
            self.blocks[position[0],position[1],min(self.l,position[2] + 1+height)] = BLOCKS_DICT['leaves']
        self.blocks[position[0],position[1],position[2]:min(self.l, position[2] + 1+height)] = BLOCKS_DICT['wood']


    def cellInDimensions(self, cell):
        for i, val in enumerate(cell):
            if val < 0 or val > self.dimensions[i]-1:
                return False
        return True

    def getAgentActions(self):
        x, y, z = self.agentPos
        adjacentCells = []
        actions = []
        if self.agentFuel > 0:
            for cell in self.getAdjacentCells():
                if self.cellInDimensions(cell):
                    actionDir = tuple(np.subtract(cell, self.agentPos))
                    # Move or place if empty
                    if self.blocks[cell[0], cell[1], cell[2]] in [0]:
                        actions.append(Action('move', cell, 1, actionDir))
                        if self.agentStorage[BLOCKS_DICT['sapling']] >= 1:
                            blockBelow = self.blocks[tuple(np.subtract(cell, (0,0,1)))]
                            if blockBelow == BLOCKS_DICT['dirt']:
                                actions.append(Action('place', (x,y,z), 1, (0,0,0), placeDirection=actionDir, itemToPlace=BLOCKS_DICT['sapling']))
                    # Break if not
                    elif self.agentFuel >= 2 and self.blocks[cell[0], cell[1], cell[2]] > 0:
                        actions.append(Action('break', cell, 2,actionDir))
        if self.agentStorage[2] > 0:
            actions.append(Action('refuel', (x,y,z), 0, (0,0,0)))
        actions.append(Action('move', (x,y,z), 0, (0,0,0)))
        return actions

    def getAdjacentCells(self):
        x, y, z = self.agentPos
        adjacentCells = []
        for step in [-1, 1]:
            adjacentCells.append((x+step, y, z))
            adjacentCells.append((x, y+step, z))
            adjacentCells.append((x, y, z+step))
        return adjacentCells
    
    def getObservation(self):
        observedObjects = []
        for i, cell in enumerate(self.getAdjacentCells()):
            if not self.cellInDimensions(cell):
                observedObjects.append(-2)
            else:
                observedObjects.append(self.blocks[cell])
        return Observation(observedObjects, deepcopy(self.agentStorage), deepcopy(self.agentFuel))

    def getHeuristicOptimalActionSeq(self):
        treeXYPos = np.dstack(np.where(self.blocks[:,:,:] == 2))
        # print(self.blocks[:,:,1])
        # print(treeXYPos)
        # print(self.agentPos)
        manhDist = np.sum(np.abs(treeXYPos - [self.agentPos[0],self.agentPos[1], self.agentPos[2]]), axis=2)[0]
        # print(manhDist)
        treeDistPair = [tuple(pair) for pair in zip(tuple(map(tuple, treeXYPos[0])), manhDist)]
        treeDistPair = sorted(treeDistPair, key=lambda x: x[1])
        # treeDistPair = np.stack((np.arange(manhDist.size), manhDist), axis=-1)
        # treeDistPair = np.sort(treeDistPair, axis=-1)
        # print(treeDistPair)
        actionSeqToGetAllWood = []
        currentPosition = self.agentPos
        currentFuel = self.agentFuel
        findTreePosition = currentPosition
        while treeDistPair:
            nextWood = treeDistPair.pop(0)
            direction = np.subtract(nextWood[0], findTreePosition)
            # fuelNeeded = nextWood[1]
            # while fuelNeeded > currentFuel:
            #     actionSeqToGetAllWood.append(Action('refuel',findTreePosition,0,(0,0,0)))
            #     currentFuel += 8

            for axisMask in [[1,0,0], [0,1,0],[0,0,1]]:
                x,y,z = findTreePosition
                distX, distY, distZ = (np.add(np.multiply(axisMask, direction), 1))
                blocksInPath = self.blocks[x:x+distX,y:y+distY,z:z+distZ].flatten()[1:]
                oldPosition = findTreePosition
                for blockType in blocksInPath:
                    newPosition = tuple(np.add(findTreePosition,axisMask))
                    if blockType == 0:
                        actionSeqToGetAllWood.append(Action('move', newPosition ,1, tuple(axisMask)))                   
                        findTreePosition = newPosition
                    elif blockType >= 1:
                        actionSeqToGetAllWood.append(Action('break', newPosition ,2, tuple(axisMask)))                   
                        findTreePosition = newPosition
                    else:
                        print(f"Can't break negative block type {blockType}")
                        raise Exception()
                    currentFuel -= actionSeqToGetAllWood[-1].cost
                    if currentFuel < 0:
                        actionSeqToGetAllWood.insert(-1,Action('refuel',oldPosition,0,(0,0,0)))
                        currentFuel += 8

        return actionSeqToGetAllWood
    

    def getTreeBaselineActions(self, numberOfTicks: int):
        actionSequence = []
        stateSequence =[]
        currenState = deepcopy(self)

        mode = 'search'
        leafRemoveCount = 0
        memory = np.full(self.blocks.shape, -1)
        while len(actionSequence) < numberOfTicks:
            stateSequence.append(currenState.__dict__)
            observation = currenState.getObservation()
            adjBlocks = list(zip(currenState.getAdjacentCells(),observation.cellContents))
            for cell, blockId in adjBlocks:
                if blockId != -2:
                    memory[cell] = blockId
                memory[currenState.agentPos] = BLOCKS_DICT['air']
                
            newAction = None
            if currenState.agentFuel <= 2:
                newAction = Action('refuel',currenState.agentPos,0,(0,0,0))
            elif mode == 'search':
                woodBlocks = np.stack(np.where(memory== BLOCKS_DICT['wood']), axis=1)
                direction = None
                if woodBlocks.size > 0:
                    totalDistance = np.subtract(woodBlocks[0], currenState.agentPos)
                    if np.sum(abs(totalDistance)) == 1:
                        mode = 'fell'
                    for i, val in enumerate(totalDistance):
                        if val != 0:
                            direction = np.zeros(3)
                            direction[i] = val // abs(val)
                else:
                    groundLevel = deepcopy(memory[:,:,1])
                    x,y = currenState.agentPos[:2]

                    xSplit = np.array_split(groundLevel, [x, x+1])
                    ySplit = np.array_split(groundLevel, [y, y+1],axis=1)
                    splits = [xSplit[0], xSplit[-1], ySplit[0], ySplit[-1]]
                    splits = [abs(np.sum(split * (split <= -1)))  for split in splits]
                    grndDirections = ((-1,0,0),(1,0,0), (0,-1,0), (0,1,0))
                    direction = grndDirections[np.argmax(splits)]
                for action in currenState.getAgentActions():
                    if all( [i == j for i, j in zip(action.posChange, direction)]):
                        newAction = action
                        break
            elif mode == 'fell':
                fellDirection = (0,0,1)
                blockAbove = tuple(np.add(fellDirection, currenState.agentPos))
                if memory[blockAbove] != BLOCKS_DICT['wood']:
                    mode = 'clearLeafs'
                    continue
                # for cell, blockId in adjBlocks:
                #     print(blockId)
                #     if cell == np.add(fellDirection, currenState.agentPos) and blockId != BLOCKS_DICT['wood']:
                #         mode = 'clearLeafs'
                #         fellDirection = (0,0,0)
                for action in currenState.getAgentActions():
                    if all( [i == j for i, j in zip(action.posChange, fellDirection)]):
                        newAction = action
                        break
            elif mode == 'clearLeafs':
                sameLevelLeaves = list(filter(lambda x: (x[1] == BLOCKS_DICT['leaves'] and x[0][2] == currenState.agentPos[2]), adjBlocks))
                if leafRemoveCount < 8 and sameLevelLeaves:
                    newPosAndBlock = random.choice(sameLevelLeaves)
                    leafRemoveCount += 1
                    for action in currenState.getAgentActions():
                        if all( [i == j for i, j in zip(action.newPosition, newPosAndBlock[0])]):
                            newAction = action
                            break
                else:
                    mode = 'plant'
                    leafRemoveCount = 0
                    continue
            elif mode == 'plant':
                grndDir = (0, 0, -1)
                blockBelow = tuple(np.add(grndDir, currenState.agentPos))
                if memory[blockBelow] != BLOCKS_DICT['dirt']:
                    for action in currenState.getAgentActions():
                        if all( [i == j for i, j in zip(action.posChange, grndDir)]):
                            newAction = action
                            break
                elif BLOCKS_DICT['sapling'] not in observation.cellContents:
                    placeActions = list(filter(lambda x: x.actionType == 'place',  currenState.getAgentActions()))
                    if placeActions:
                        assert(currenState.agentStorage[BLOCKS_DICT['sapling']] > 0)
                        newAction = random.choice(placeActions)
                    else:
                        mode = 'search'
                        continue
                else:
                    # Waiting
                    if BLOCKS_DICT['wood'] in observation.cellContents:
                        mode = 'search'
                        continue
                    waitAction = list(filter(lambda x: x.posChange == (0,0,0),  currenState.getAgentActions()))
                    newAction = waitAction[0]
            else:
                print("Not found")

            assert(newAction != None)
            currenState = currenState.executeAction(newAction)
            actionSequence.append(newAction)

        return actionSequence, stateSequence

    def reward(self):
        return self.agentStorage[2]

    def removeTreeFromGrowList(self, treePosition):
        for treeGrowing in self.treeToGrow:
            if treeGrowing.position == treePosition:
                self.treeToGrow.remove(treeGrowing)
        return

    def executeAction(self, action:Action):
        newState = deepcopy(self)
        newState.agentPos = action.newPosition

        # Sufficient fuel for the action
        newState.agentFuel -= action.cost
        assert(newState.agentFuel >= 0)
        if action.actionType == "move":
            newState.blocks[self.agentPos] = 0
        elif action.actionType == "break":
            blockBroken = self.blocks[action.newPosition]
            blockAbovePos = tuple(np.add(action.newPosition, (0,0,1)))
            newState.blocks[self.agentPos] = 0

            # Sapling logic
            saplingDropRate = 0.3
            if blockBroken == BLOCKS_DICT['leaves'] and saplingDropRate >= np.random.uniform():
                newState.agentStorage[BLOCKS_DICT['sapling']] += 1
            # Remove sapling above broken block
            elif blockAbovePos[2] < newState.l and self.blocks[blockAbovePos] == BLOCKS_DICT['sapling']:
                newState.blocks[blockAbovePos] = BLOCKS_DICT['air']
                newState.agentStorage[self.blocks[blockAbovePos]] += 1
                newState.removeTreeFromGrowList(blockAbovePos)
            elif blockBroken == BLOCKS_DICT['sapling']:
                newState.removeTreeFromGrowList(action.newPosition)
                
            newState.agentStorage[blockBroken] += 1
        elif action.actionType == "refuel":
            assert(newState.agentStorage[2] > 0)
            newState.agentFuel += 8
            newState.agentStorage[2] -= 1
        elif action.actionType == "place":
            placePosition = tuple(np.add(action.placeDirection, action.newPosition))
            if action.itemToPlace == BLOCKS_DICT['sapling']:
                blockBelow = newState.blocks[tuple(np.subtract(placePosition, (0,0,1)))]
                assert(blockBelow == BLOCKS_DICT['dirt'])
                newState.treeToGrow.append(TreeGrowTime(placePosition, newState.worldTick + 8))
            assert(newState.agentStorage[action.itemToPlace] >= 1)
            newState.blocks[placePosition] = action.itemToPlace
            newState.agentStorage[action.itemToPlace] -= 1
        else:
            error(f"Action {actionType} not defined")
        newState.blocks[action.newPosition] = -1
        assert(newState.blocks[newState.agentPos] == -1)
        # Update time tick
        newState.worldTick += 1
        # Grow trees:
        newState.treeToGrow = sorted(newState.treeToGrow, key=lambda x: x.growthDoneTick)

        while len(newState.treeToGrow) > 0 and newState.treeToGrow[0].growthDoneTick < newState.worldTick:
            treeGrowInfo = newState.treeToGrow.pop(0)
            newState.plantTree(treeGrowInfo.position)
            
        return newState

    

    def heuristicDfs(self):
        root = SearchNode(self)
        stack = [root]
        visited = set() 

        while stack:
            node = stack.pop()  # Pop the last node from the stack

            if node.state == goal_state:
                return node.action

            visited.add(node.state)

            # Add unvisited children to the stack
            for child in node.children:
                if child.state not in visited:
                    stack.append(child)

        # Goal state not found
        return None

    def moveAgentTo(self, x, y, z):
        newState = deepcopy(self)
        indices = np.where(self.blocks == -1)
        x_indices, y_indices, z_indices = indices
        newState.blocks[x_indices, y_indices, z_indices] = 0
        newState.blocks[x, y, z] = -1
        newState.agentPos = (x, y, z)


    def print(self):
        for layer in range(self.l):
            print(f"Layer {layer}:")
            for row in range(self.h):
                for col in range(self.w):
                    print(int(self.blocks[col, row, layer]), end=" ")
                print()
            print()


h = 6  # Height of the world
w = 6  # Width of the world
l = 8  # Number of layers

# myWorld = World(h, w, l)
# myWorld.generateTreeWorld()
# myWorld.print()
# actions, states = myWorld.getTreeBaselineActions(200)

# print(len(actions))
# for i, (action, state) in enumerate(zip(actions, states)):
#     print(i)
#     print(state.agentStorage)
#     print(action)
#     assert(myWorld.agentStorage == state.agentStorage)
#     myWorld = myWorld.executeAction(action)
#     print(myWorld.agentFuel)
#     print(myWorld.agentStorage)

# print(len(actions))

# stateActionPairs = [{'state': s, 'action': a} for s, a in zip(states,actions)]
# with open(baselinePath, 'w') as file:
#     json.dump(stateActionPairs, file, cls=NpEncoder)



# myWorld.agentFuel = 10
# myWorld.agentStorage[BLOCKS_DICT['sapling']] = 1
# actions = myWorld.getAgentActions()
# for action in actions:
#     if action.actionType == 'place':
#         break

# myWorld.print()