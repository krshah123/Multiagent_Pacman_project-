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

        "*** YOUR CODE HERE ***"
        foods = newFood.asList()
        aim = 0
        if successorGameState.isWin():
            return float("inf")
        for y in newGhostStates:
            if y.scaredTimer == 0:

                distance = manhattanDistance(y.getPosition(), newPos)
                if distance <= 1:
                    return float("-inf") 

            if y.scaredTimer > 0:
                if manhattanDistance(y.getPosition(), newPos) <= 1:
                    aim = aim + 95    
        
        if action == 'Stop':
            aim = aim - 40 

        food_distance = []
        for food in foods:
            distance = manhattanDistance(food, newPos)
            food_distance.append(distance)

        min_distance = min(food_distance)

        if min_distance > 0:
            aim = aim + 1 / min_distance

        if food:
            aim = aim - len(foods)

        return aim
        return successorGameState.getScore()

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
        totalghosts = gameState.getNumAgents() - 1

      
        def pmax(gameState,depth):
            current1 = depth + 1 
            if gameState.isWin(): 
                return self.evaluationFunction(gameState)
                
            if gameState.isLose():
                return self.evaluationFunction(gameState)
            
            if current1==self.depth:   
                return self.evaluationFunction(gameState)


            max_v = float("-inf") 
            
            actions = []
            actions.extend(gameState.getLegalActions(0))

            for x in actions:
                next_state = gameState.generateSuccessor(0, x)

                value = gmin(next_state, current1, 1)

                if value > max_v:
                    max_v = value 

            return max_v
        
  
        def gmin(gameState,d, turn):
            min_v = float("inf")	


            if gameState.isWin():
                return scoreEvaluationFunction(gameState)

            if gameState.isLose():
                return scoreEvaluationFunction(gameState)
            
            actions = []
            actions.extend(gameState.getLegalActions(turn))

            for x in actions:
                next_state = gameState.generateSuccessor(turn,x) 


                if turn == (gameState.getNumAgents() - 1):
                    min_value1 = pmax(next_state, d)

                    if min_value1 < min_v:
                        min_v = min_value1
                else:
                    min_value1 = gmin(next_state, d, turn + 1)

                    min_v = min(min_v, min_value1)


            return min_v
        

        actions = []

        actions.extend(gameState.getLegalActions(0))

        cur = float("-inf")

        action1 = str()

        for x in actions:
            nextState = gameState.generateSuccessor(0,x) 

            points = gmin(nextState,0,1)
            if points > cur:
            
                action1 = x
                cur = points

            else:
                action1 = action1
                cur = cur

        return action1

        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"      
        def pmax(State,depth,alpha, beta):

            current3 = 1 + depth

            if State.isWin():
                return self.evaluationFunction(State)
            if State.isLose(): 
                return self.evaluationFunction(State)
            if current3==self.depth:  
                return self.evaluationFunction(State)

            high = float("-inf") 

            actions = []
            actions.extend(State.getLegalActions(0))


            for x in actions:
                successor= State.generateSuccessor(0, x) 

                new = gmin(successor, current3, 1, alpha, beta)

                if new > high:
                    high = new


                if high > beta:
                    return high
  
                new_alpha = max(alpha, high)
                alpha = new_alpha

            return high
        
       
        def gmin(State, depth, ind, alpha, b):

            minimum = float("inf") 

            if State.isWin():
                return self.evaluationFunction(State)
            if State.isLose(): 
                return self.evaluationFunction(State)
            
            actions = []
            actions.extend(State.getLegalActions(ind))

            for x in actions:
                next_state1= State.generateSuccessor(ind, x) 


                if ind == State.getNumAgents() - 1:

                    new_minimum = pmax(next_state1, depth, alpha, b)
                    minimum = min(minimum, new_minimum)


                    if alpha > minimum:
                        return minimum

                    b = min(b, minimum)
                else:
                    minimum = min(minimum,gmin(next_state1, depth, ind+1 , alpha, b))

                    if minimum < alpha: 
                        return minimum

                    b = min(b,minimum)
            return minimum


        actions = []
        actions.extend(gameState.getLegalActions(0))

        original = float("-inf") 
        best1 = str()

        alpha, beta = float("-inf"), float("inf")

        for x in actions:
            nextState = gameState.generateSuccessor(0,x)
    

            new = gmin(nextState, 0, 1, alpha, beta)
            score = new

            if original < score:

                best1 = x
                original = score


            if score > beta:
                return best1
            alpha = max(alpha, score)

        return best1		

        util.raiseNotDefined()


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
        def pmax(State,depth):
            current = depth + 1 

            if State.isWin():
                return self.evaluationFunction(State)

            if State.isLose():
                return self.evaluationFunction(State)

            if current==self.depth:   
                return self.evaluationFunction(State)

            max_val = float("-inf")

            actions = []
            actions.extend(State.getLegalActions(0))
            
            totalmaxvalue = 0

            num = len(actions)
    


            for x in actions:
                next_state = State.generateSuccessor(0, x)

                exp_val = expectLevel(next_state, current, 1)

                max_val = max(max_val, exp_val)

            return max_val
        

        def expectLevel(State, depth, turn):

            if State.isWin():
                return self.evaluationFunction(State)
            if State.isLose():  
                return self.evaluationFunction(State)

            val1  = 0

            actions = []
            actions.extend(State.getLegalActions(turn))
           

            num = len(actions)

            for x in actions:
                next_state= State.generateSuccessor(turn,x) 

                num_agents = State.getNumAgents()

                if turn == num_agents - 1:
                    val = pmax(next_state, depth)

                else:
                    val = expectLevel(next_state,depth,turn+1)

                val1 = val1 + val


            if num == 0:
                result = 0

            else:
                result = float(val1) / num

            return result
        
        
        actions = []

        actions.extend(gameState.getLegalActions(0))
        

        current = float("-inf")

        best = str()

        turn1 = 0

        for x in actions:
            nextState = gameState.generateSuccessor(turn1,x)
            
            turn2 = 0
            depth1 = 1

            point = expectLevel(nextState, turn2, depth1)
            


            if point > current:
                best = x
                current = point

            else:
                best = best 
                current = current  

        return best

        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    Position = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    States1 = currentGameState.getGhostStates()

    List = Food.asList()

    Food = 9000000 

    entity = 9000000 


    for x in States1:
        X, Y = x.getPosition()

        if x.scaredTimer == 0:

            distance = manhattanDistance((X, Y), Position)

            entity = min(entity, distance)

    
    currents = currentGameState.getScore()

    score_5 = currents - 5
    entity_s = entity + 1
    entity_1 = 1.0 / entity_s
    half_food = Food / 2


    hero = (score_5 * entity_1) - half_food

    return hero

    
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
