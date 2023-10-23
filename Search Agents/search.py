# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    from util import Queue
    from util import Stack
    start_node =problem.getStartState()
    visited = Stack()
    frontier = Stack()
    item = start_node
    frontier.push(item)
    "dictionary paren child relation"
    parent_child_dictionary = {}
    if problem.isGoalState(start_node):
        return [];
    while True :
        if frontier.isEmpty() :
            return [];
        node = frontier.pop()
        visited.push(node)
        if problem.isGoalState(node):
            roadlist = []
            n = node
            while True:
                if n == start_node:
                    return list(reversed(roadlist))
                par,act=parent_child_dictionary[n]
                n= par
                roadlist.append(act)
        successors = problem.getSuccessors(node)
        for nextState, action, cost in successors :
            child = nextState
            if not ((child in visited.list)):
                parent_child_dictionary[child] = (node,action)
                frontier.push(child)
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    from util import Queue
    start_node =problem.getStartState()
    visited = Queue()
    frontier = Queue()
    item = start_node
    frontier.push(item)
    parent_child_dictionary = {}
    if problem.isGoalState(start_node):
        return [];
    while True :
        if frontier.isEmpty() :
            return [];
        node = frontier.pop()
        if problem.isGoalState(node):
            roadlist = []
            n = node
            while True:
                if n == start_node:
                    return list(reversed(roadlist))
                par,act=parent_child_dictionary[n]
                n= par
                roadlist.append(act)
        visited.push(node)
        successors = problem.getSuccessors(node)
        for nextState, action, cost in successors :
            child = nextState
            if not ((child in visited.list ) or (child in frontier.list)):
                parent_child_dictionary[child] = (node,action)
                frontier.push(child)






    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    from util import Queue
    start_node =problem.getStartState()
    visited = Queue()
    frontier = PriorityQueue()
    item = start_node
    frontier.push(item,0)
    "dictionary paren child relation"
    parent_child_dictionary = {}
    if problem.isGoalState(start_node):
        return [];
    while True :
        if frontier.isEmpty() :
            return [];
        node = frontier.pop()
        if problem.isGoalState(node):
            roadlist = []
            n = node
            while True:
                if n == start_node:
                    return list(reversed(roadlist))
                par,act=parent_child_dictionary[n]
                n= par
                roadlist.append(act)
        visited.push(node)
        successors = problem.getSuccessors(node)
        for nextState, action, cost in successors :
            child = nextState
            if not ((child in visited.list ) or child in [dot for sublist in frontier.heap for dot in sublist] ):
                parent_child_dictionary[child] = (node,action)
                roadlist = []
                n= child
                while True:
                    if n == start_node:
                        break
                    par,act=parent_child_dictionary[n]
                    n= par
                    roadlist.append(act)
                act=problem.getCostOfActions(list(reversed(roadlist)))
                frontier.update(child,act)
            elif child in [dot for sublist in frontier.heap for dot in sublist]:
                roadlist = []
                n = child
                keep1,keep2 = parent_child_dictionary[n]
                while True:
                    if n == start_node:
                        break
                    par,act=parent_child_dictionary[n]
                    n= par
                    roadlist.append(act)
                cost_of_old =problem.getCostOfActions(list(reversed(roadlist)))
                "new cost"
                roadlist = []
                child2 = child
                parent_child_dictionary[child2] = (node,action)
                n  = child2
                while True:
                    if n == start_node:
                        break
                    par,act=parent_child_dictionary[n]
                    n= par
                    roadlist.append(act)
                cost_of_new =problem.getCostOfActions(list(reversed(roadlist)))
                if cost_of_new<cost_of_old:
                    frontier.update(child,cost_of_new)
                else:
                    parent_child_dictionary[child] = (keep1,keep2)









    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    from util import Queue
    start_node =problem.getStartState()
    visited = Queue()
    frontier = PriorityQueue()
    item = start_node
    frontier.push(item,0)
    parent_child_dictionary = {}
    if problem.isGoalState(start_node):
        return [];
    while True :
        if frontier.isEmpty() :
            return [];
        node = frontier.pop()
        if problem.isGoalState(node):
            roadlist = []
            n = node
            while True:
                if n == start_node:
                    return list(reversed(roadlist))
                par,act=parent_child_dictionary[n]
                n= par
                roadlist.append(act)
        visited.push(node)
        successors = problem.getSuccessors(node)
        for nextState, action, cost in successors :
            child = nextState
            if not ((child in visited.list ) or child in [dot for sublist in frontier.heap for dot in sublist] ):
                parent_child_dictionary[child] = (node,action)
                roadlist = []
                n= child
                while True:
                    if n == start_node:
                        break
                    par,act=parent_child_dictionary[n]
                    n= par
                    roadlist.append(act)
                act=problem.getCostOfActions(list(reversed(roadlist)))
                frontier.update(child,act +heuristic(child,problem) )
            elif child in [dot for sublist in frontier.heap for dot in sublist]:
                roadlist = []
                n = child
                keep1,keep2 = parent_child_dictionary[n]
                while True:
                    if n == start_node:
                        break
                    par,act=parent_child_dictionary[n]
                    n= par
                    roadlist.append(act)
                cost_of_old =problem.getCostOfActions(list(reversed(roadlist)))
                "new cost"
                roadlist = []
                child2 = child
                parent_child_dictionary[child2] = (node,action)
                n  = child2
                while True:
                    if n == start_node:
                        break
                    par,act=parent_child_dictionary[n]
                    n= par
                    roadlist.append(act)
                cost_of_new =problem.getCostOfActions(list(reversed(roadlist)))
                if cost_of_new<cost_of_old:
                    frontier.update(child,cost_of_new + heuristic(child,problem))
                else:
                    parent_child_dictionary[child] = (keep1,keep2)





    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
