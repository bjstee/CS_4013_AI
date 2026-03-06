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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        walls = successorGameState.getWalls()

        "*** YOUR CODE HERE ***"
        """
        Evaluation function:
        - Strart with score of successor state
        - Subtract distance to closest food
        - Subtract sum of distances to ghosts

        """
        
        score = successorGameState.getScore() # Start with the score of the successor state
        foodList = newFood.asList() # Get the list of food positions
        if foodList:
            closestFoodDist = min(manhattanDistance(newPos, food) for food in foodList)
            score += 1 / closestFoodDist # Subtract distance to closest food

        for ghost in newGhostStates:
            ghostDist = manhattanDistance(newPos, ghost.getPosition())
            if newScaredTimes[0] > 0:
                score += 10 / (ghostDist + 1) # Add score for scared ghosts
            else:
                if ghostDist < 3:
                    score -= 100 # penalize heavily for being too close
                else:
                    score -= 2 / (ghostDist + 1) # Subtract score for non-scared ghosts

        score -= len(foodList) * 4 # Penalize number of remaining food (multiply by 4 for greater effect)

        if action == Directions.STOP:
            score -= 5 # Penalize stopping
        
        return score


def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        def minimax(state, depth, agentIndex):
            # Terminal conditions
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            nextAgent = agentIndex + 1
            if nextAgent == state.getNumAgents():
                nextAgent = 0
            nextDepth = depth - 1 if nextAgent == 0 else depth

            # pacman wants to maximize the score
            if agentIndex == 0:
                bestScore = float('-inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action) 
                    score = minimax(successor, nextDepth, nextAgent)
                    if score > bestScore:
                        bestScore = score
                return bestScore

            # ghosts want to minimize the score
            else:
                bestScore = float('inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    score = minimax(successor, nextDepth, nextAgent)
                    if score < bestScore:
                        bestScore = score
                return bestScore

        # try every action pacman can take and pick the best one
        bestAction = None
        bestScore = float('-inf')

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = minimax(successor, self.depth, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBetaMinimax(state, depth, agentIndex, alpha, beta):

            def max_value(state, depth, agentIndex, alpha, beta):
                bestScore = float('-inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action) 
                    score = alphaBetaMinimax(successor, nextDepth, nextAgent, alpha, beta)
                    bestScore = max(bestScore, score)
                    if bestScore > beta:
                        return bestScore  # prune here
                    alpha = max(alpha, bestScore)
                return bestScore

            def min_value(state, depth, agentIndex, alpha, beta):
                bestScore = float('inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    score = alphaBetaMinimax(successor, nextDepth, nextAgent, alpha, beta)
                    bestScore = min(bestScore, score)
                    if bestScore < alpha:
                        return bestScore   # prune here
                    beta = min(beta, bestScore)
                return bestScore

            # Terminal conditions
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth - 1 if nextAgent == 0 else depth

            if agentIndex == 0:
                return max_value(state, depth, agentIndex, alpha, beta)
            else:
                return min_value(state, depth, agentIndex, alpha, beta)

        # try every action pacman can take and pick the best one
        bestAction = None
        bestScore = float('-inf')
        alpha = float('-inf')

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = alphaBetaMinimax(successor, self.depth, 1, alpha, float('inf'))
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, depth, agentIndex):
            # Terminal conditions
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth - 1 if nextAgent == 0 else depth

            if agentIndex == 0: # pacman maximizes score
                bestScore = float('-inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action) 
                    score = expectimax(successor, nextDepth, nextAgent)
                    bestScore = max(bestScore, score)
                return bestScore
            else: # compute average score
                totalScore = 0
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    score = expectimax(successor, nextDepth, nextAgent)
                    totalScore += score
                return totalScore / len(legalActions) if legalActions else 0
        
        # try every action pacman can take and pick the best one
        bestAction = None
        bestScore = float('-inf')

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = expectimax(successor, self.depth, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    This evaluation function is the same as Q1 above.
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore() # Start with the score of the current state
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList() # Get the list of food
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    if foodList:
        closestFoodDist = min(manhattanDistance(pacmanPos, food) for food in foodList)
        score += 1 / closestFoodDist # Add score for being closer to food
    for ghost in ghostStates:
        ghostDist = manhattanDistance(pacmanPos, ghost.getPosition())
        if scaredTimes[0] > 0:
            score += 10 / (ghostDist + 1) # Add score for scared ghosts
        else:
            if ghostDist < 3:
                score -= 100 # penalize heavily for being too close
            else:
                score -= 2 / (ghostDist + 1) # Subtract score for non-scared ghosts
    score -= len(foodList) * 4 # Penalize number of remaining food (multiply by 4 for greater effect)
    return score
    
# Abbreviation
better = betterEvaluationFunction
