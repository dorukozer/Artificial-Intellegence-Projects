# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import numpy as np

from util import manhattanDistance
from game import Directions
import random, util
import numpy

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        score =successorGameState.getScore()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()


        #extracts ghosts and saves them to maps
        #first map is the position of the ghost
        #second map is schecking if the ghost is scared or not
        ghosts= {}
        ghosts_scared= {}
        for idx,x in enumerate(newGhostStates):
            ghosts[idx] = x.getPosition()
            ghosts_scared[idx] = x.scaredTimer > 0


        #punishes the pacman with factors
        #if the unscared ghost is in 3 manhattan distance it is punished 55 2 manhattan distance 65 ....

        mostDanger= 3
        num_sc= 0
        factor = [100,75,65,55]
        for key in ghosts:
            mantoGhost = manhattanDistance(ghosts[key],newPos)
            isScared=ghosts_scared[key]
            if (isScared != True):
                if mantoGhost<mostDanger:
                    mostDanger = mantoGhost
            else:
                num_sc+=1



        """
        # this map is created for tracking the location of the capsules
        caps_map= {}
        capsules=successorGameState.getCapsules()
        for idx,x in enumerate(capsules):
            caps_map[idx] = x
        
        #here the main goals is to find the closest manhattan distance to caposule
        # this works same logic like I implemented for ghosts
        # I will be  awarding  more if pacman eats the capsule
        leastDanger= 2
        factor2 = [100,75,65]
        for key in caps_map:
            mantoCaps= manhattanDistance(caps_map[key],newPos)
            if mantoCaps<leastDanger:
                leastDanger = int(mantoCaps)
        """
        manhDist = {}

        #here I found the closest manhattan distance to the closest food in order to go there
        j=0
        for y in range(newFood.height):
            for x in range(newFood.width):
                if newFood[x][y] == True:
                    manhDist[j]=manhattanDistance(newPos,(x,y))
                    j=j+1


        if(len(manhDist) !=0):
            dictionary_keys = list(manhDist.keys())
            sorted_dict = {dictionary_keys[i]: sorted(manhDist.values())[i] for i in range(len(dictionary_keys))}
            iterable = iter(sorted_dict.items())
            ke,va = next(iterable)
            #to punish the length of the closest food
            score -= int(va)/12
        # to not punish the score if the ghost is scared
        if num_sc ==0:
            score =score- 3*factor[int(mostDanger)]

        #score =score+factor2[int(leastDanger)]



        "*** YOUR CODE HERE ***"
        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.get_best_action(gameState,self.depth,0)




    def get_best_action(self ,startState,depth,turn):
        legalMoves = startState.getLegalActions()
        max = -99999999999
        best_act=None
        for idx, action in enumerate(legalMoves):
            succState=startState.generateSuccessor(turn, action)
            val=self.value(succState,depth,1)
            if val >max:
                best_act = action
                max= val
        return best_act

    def value(self,state,depth,turn):
        if depth == 0 or state.isLose() or state.isWin():
            return self.evaluationFunction(state)
        if turn ==0:
            return self.max_value(state,depth,turn)
        else:
            return self.min_value(state,depth,turn)


    def min_value(self,state,depth,turn):
        v = 9999999999999999
        next_turn = turn
        next_depth= depth
        legalMoves = state.getLegalActions(turn)
        if turn == state.getNumAgents() - 1:
            next_turn =0
            next_depth = depth - 1
        else:
            next_turn =turn + 1
            next_depth = depth
        for idx, action in enumerate(legalMoves):
            succState= state.generateSuccessor(turn, action)
            v = min(v,self.value(succState,next_depth,next_turn))
        return  v

    def max_value(self,state,depth,turn):
        v = -9999999999999999
        next_turn = turn
        next_depth= depth
        legalMoves = state.getLegalActions(turn)
        if turn == state.getNumAgents() - 1:
            next_turn =0
            next_depth = depth - 1
        else:
            next_turn =turn + 1
            next_depth = depth
        for idx, action in enumerate(legalMoves):
            succState=state.generateSuccessor(turn, action)
            v = max(v,self.value(succState,next_depth,next_turn))
        return  v






class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.get_best_action(gameState,self.depth,0,-9999999999999999,9999999999999999)


    def get_best_action(self ,startState,depth,turn,alpha,beta):
        legalMoves = startState.getLegalActions()
        max_ = -9999999999999999
        best_act=None
        next_turn = turn
        next_depth= depth
        next_alpha = alpha
        next_beta=beta
        if turn == startState.getNumAgents() - 1:
            next_turn =0
            next_depth = depth - 1
        else:
            next_turn =turn + 1
            next_depth = depth

        for idx, action in enumerate(legalMoves):
            succState=startState.generateSuccessor(turn, action)
            val=self.value(succState,next_depth,next_turn,next_alpha,next_beta)
            if val >max_:
                best_act = action
                max_= val
            if val> next_beta:
                return best_act
            next_alpha = max(val,next_alpha)
        return best_act

    def value(self,state,depth,turn,alpha,beta):
        if depth == 0 or state.isLose() or state.isWin():
            return self.evaluationFunction(state)
        if turn ==0:
            return self.max_value(state,depth,turn, alpha, beta)
        else:
            return self.min_value(state,depth,turn, alpha, beta)


    def min_value(self,state,depth,turn, alpha, beta):
        v = 9999999999999999
        next_turn = turn
        next_depth= depth

        next_beta=beta
        legalMoves = state.getLegalActions(turn)
        if turn == state.getNumAgents() - 1:
            next_turn =0
            next_depth = depth - 1
        else:
            next_turn =turn + 1
            next_depth = depth
        for idx, action in enumerate(legalMoves):
            succState= state.generateSuccessor(turn, action)
            v = min(v,self.value(succState,next_depth,next_turn, alpha, next_beta))
            if v< alpha:
                return v
            next_beta = min(next_beta,v)
        return  v

    def max_value(self,state,depth,turn, alpha, beta):
        v = -9999999999999999
        next_turn = turn
        next_depth= depth
        next_alpha = alpha
        legalMoves = state.getLegalActions(turn)
        if turn == state.getNumAgents() - 1:
            next_turn =0
            next_depth = depth - 1
        else:
            next_turn =turn + 1
            next_depth = depth

        for idx, action in enumerate(legalMoves):
            succState = state.generateSuccessor(turn, action)
            v = max(v,self.value(succState,next_depth,next_turn, next_alpha, beta))
            if v >beta:
                return v
            next_alpha = max(v,alpha)
        return  v


        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.get_best_action(gameState,self.depth,0)




    def get_best_action(self ,startState,depth,turn):
        legalMoves = startState.getLegalActions()
        max = -99999999999
        best_act=None
        for idx, action in enumerate(legalMoves):
            succState=startState.generateSuccessor(turn, action)
            val=self.value(succState,depth,1)
            if val >max:
                best_act = action
                max= val
        return best_act

    def value(self,state,depth,turn):
        if depth == 0 or state.isLose() or state.isWin():
            return self.evaluationFunction(state)
        if turn ==0:
            return self.max_value(state,depth,turn)
        else:
            return self.min_value(state,depth,turn)



    def min_value(self,state,depth,turn):
        v = 0
        next_turn = turn
        next_depth= depth
        legalMoves = state.getLegalActions(turn)
        if turn == state.getNumAgents() - 1:
            next_turn =0
            next_depth = depth - 1
        else:
            next_turn =turn + 1
            next_depth = depth
        for idx, action in enumerate(legalMoves):
            succState= state.generateSuccessor(turn, action)
            v += self.value(succState,next_depth,next_turn)
        return  v/len(legalMoves)

    def max_value(self,state,depth,turn):
        v = -9999999999999999
        next_turn = turn
        next_depth= depth
        legalMoves = state.getLegalActions(turn)
        if turn == state.getNumAgents() - 1:
            next_turn =0
            next_depth = depth - 1
        else:
            next_turn =turn + 1
            next_depth = depth
        for idx, action in enumerate(legalMoves):
            succState=state.generateSuccessor(turn, action)
            v = max(v,self.value(succState,next_depth,next_turn))
        return  v


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    score =currentGameState.getScore()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()


    #extracts ghosts and saves them to maps
    #first map is the position of the ghost
    #second map is schecking if the ghost is scared or not
    ghosts= {}
    ghosts_scared= {}
    for idx,x in enumerate(newGhostStates):
        ghosts[idx] = x.getPosition()
        ghosts_scared[idx] = x.scaredTimer > 0


    #punishes the pacman with factors
    #if the unscared ghost is in 3 manhattan distance it is punished 55 2 manhattan distance 65 ....
    mostDanger= 3
    num_sc= 0
    factor = [100,75,65,55]
    for key in ghosts:
        mantoGhost = manhattanDistance(ghosts[key],newPos)
        isScared=ghosts_scared[key]
        if (isScared != True):
            if mantoGhost<mostDanger:
                mostDanger = mantoGhost
        else:
            num_sc+=1




    # this map is created for tracking the location of the capsules
    caps_map= {}
    capsules=currentGameState.getCapsules()
    for idx,x in enumerate(capsules):
        caps_map[idx] = x

    #here the main goals is to find the closest manhattan distance to caposule
    # this works same logic like I implemented for ghosts
    # I will be  awarding  more if pacman eats the capsule
    leastDanger= 2
    factor2 = [100,75,65]
    for key in caps_map:
        mantoCaps= manhattanDistance(caps_map[key],newPos)
        if mantoCaps<leastDanger:
            leastDanger = int(mantoCaps)
    manhDist = {}

    #here I found the closest manhattan distance to the closest food in order to go there
    j=0
    for y in range(newFood.height):
        for x in range(newFood.width):
            if newFood[x][y] == True:
                manhDist[j]=manhattanDistance(newPos,(x,y))
                j=j+1


    if(len(manhDist) !=0):
        dictionary_keys = list(manhDist.keys())
        sorted_dict = {dictionary_keys[i]: sorted(manhDist.values())[i] for i in range(len(dictionary_keys))}
        iterable = iter(sorted_dict.items())
        ke,va = next(iterable)
        # in order to go and eat the most closest food and go to that direction
        score -= int(va)/12
    # this if statement only punsihes the state to make pacman go away from ghost
    if num_sc ==0:
        score =score- 3*factor[int(mostDanger)]
    # this if statment awards the score if the food is in 3 manhattan distance and increases as it closer to the food
    score =score+factor2[int(leastDanger)]


    return score


    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


