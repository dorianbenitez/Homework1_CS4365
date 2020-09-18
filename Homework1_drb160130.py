#######
# File: Homework1_drb160130.py
# Author: Dorian Benitez (drb160130)
# Date: 9/18/2020
# Purpose: CS 4365.501 - Homework #1 (Search Algorithms)
#######

import sys
from collections import deque
from heapq import heappush, heappop, heapify
import itertools

# Class to update the game state
class State:

    def __init__(self, state, parent, move, depth, cost, key):
        self.state = state
        self.parent = parent
        self.move = move
        self.depth = depth
        self.cost = cost
        self.key = key

        if self.state:
            self.map = ''.join(str(e) for e in self.state)

    def __eq__(self, other):
        return self.map == other.map

    def __lt__(self, other):
        return self.map < other.map


# Create global variables that will remain as constant values
goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
goal_node = State
initial_state = list()
board_len = 0
board_side = 0
max_search_depth = 10
enqueued_states = 0
moves = list()
costs = set()


# Function to perform Breadth-First-Search
def bfs(start_state):
    global enqueued_states, goal_node, max_search_depth

    explored, queue = set(), deque([State(start_state, None, None, 0, 0, 0)])

    # Perform a BFS operation on the current queue
    while queue:
        node = queue.popleft()
        explored.add(node.map)

        if node.state == goal_state:
            goal_node = node
            return queue

        neighbors = expand(node)

        for neighbor in neighbors:
            if neighbor.map not in explored:
                queue.append(neighbor)
                explored.add(neighbor.map)

                # The program ends when the search depth exceeds 10
                if neighbor.depth > max_search_depth:
                    print("The search depth exceeded 10. Ending program...")
                    exit(0)

        if len(queue) > enqueued_states:
            enqueued_states = len(queue)


# Function to perform an Iterative Deepening Search
def ids(start_state):
    global costs

    threshold = h1(start_state)

    while 1:
        response = dls_mod(start_state, threshold)

        if type(response) is list:
            return response
            break

        threshold = response
        costs = set()

# Function to perform the operations behind the Iterative Deepening Search
def dls_mod(start_state, threshold):
    global enqueued_states, goal_node, max_search_depth, costs

    explored, stack = set(), list([State(start_state, None, None, 0, 0, threshold)])

    while stack:
        node = stack.pop()
        explored.add(node.map)

        if node.state == goal_state:
            goal_node = node
            return stack

        if node.key > threshold:
            costs.add(node.key)

        if node.depth < threshold:
            neighbors = reversed(expand(node))

            for neighbor in neighbors:
                if neighbor.map not in explored:
                    neighbor.key = neighbor.cost + h1(neighbor.state)
                    stack.append(neighbor)
                    explored.add(neighbor.map)

                    if neighbor.depth > max_search_depth:
                        print("The search depth exceeded 10. Ending program...")
                        exit(0)

            if len(stack) > enqueued_states:
                enqueued_states = len(stack)

    return min(costs)

# First A* search using the first heuristic
def astar1(start_state):
    global enqueued_states, goal_node, max_search_depth

    explored, heap, heap_entry, counter = set(), list(), {}, itertools.count()
    key = h1(start_state)
    root = State(start_state, None, None, 0, 0, key)
    entry = (key, 0, root)
    heappush(heap, entry)
    heap_entry[root.map] = entry

    while heap:
        node = heappop(heap)
        explored.add(node[2].map)

        if node[2].state == goal_state:
            goal_node = node[2]
            return heap

        neighbors = expand(node[2])

        for neighbor in neighbors:
            neighbor.key = neighbor.cost + h1(neighbor.state)
            entry = (neighbor.key, neighbor.move, neighbor)

            if neighbor.map not in explored:
                heappush(heap, entry)
                explored.add(neighbor.map)
                heap_entry[neighbor.map] = entry

                if neighbor.depth > max_search_depth:
                    print("The search depth exceeded 10. Ending program...")
                    exit(0)

            elif neighbor.map in heap_entry and neighbor.key < heap_entry[neighbor.map][2].key:

                hindex = heap.index((heap_entry[neighbor.map][2].key,
                                     heap_entry[neighbor.map][2].move,
                                     heap_entry[neighbor.map][2]))

                heap[int(hindex)] = entry
                heap_entry[neighbor.map] = entry
                heapify(heap)

        if len(heap) > enqueued_states:
            enqueued_states = len(heap)


# Second A* search using the second heuristic
def astar2(start_state):
    global enqueued_states, goal_node, max_search_depth

    explored, heap, heap_entry, counter = set(), list(), {}, itertools.count()
    key = h2(start_state)
    root = State(start_state, None, None, 0, 0, key)
    entry = (key, 0, root)
    heappush(heap, entry)
    heap_entry[root.map] = entry

    while heap:
        node = heappop(heap)
        explored.add(node[2].map)

        if node[2].state == goal_state:
            goal_node = node[2]
            return heap

        neighbors = expand(node[2])

        for neighbor in neighbors:
            neighbor.key = neighbor.cost + h2(neighbor.state)
            entry = (neighbor.key, neighbor.move, neighbor)

            if neighbor.map not in explored:
                heappush(heap, entry)
                explored.add(neighbor.map)
                heap_entry[neighbor.map] = entry

                if neighbor.depth > max_search_depth:
                    print("The search depth exceeded 10. Ending program...")
                    exit(0)

            elif neighbor.map in heap_entry and neighbor.key < heap_entry[neighbor.map][2].key:

                hindex = heap.index((heap_entry[neighbor.map][2].key,
                                     heap_entry[neighbor.map][2].move,
                                     heap_entry[neighbor.map][2]))

                heap[int(hindex)] = entry
                heap_entry[neighbor.map] = entry
                heapify(heap)

        if len(heap) > enqueued_states:
            enqueued_states = len(heap)

# Function to return the neighbor nodes of the node being passed
def expand(node):
    neighbors = list()

    neighbors.append(State(move(node.state, 1), node, 1, node.depth + 1, node.cost + 1, 0))
    neighbors.append(State(move(node.state, 2), node, 2, node.depth + 1, node.cost + 1, 0))
    neighbors.append(State(move(node.state, 3), node, 3, node.depth + 1, node.cost + 1, 0))
    neighbors.append(State(move(node.state, 4), node, 4, node.depth + 1, node.cost + 1, 0))

    nodes = [neighbor for neighbor in neighbors if neighbor.state]

    return nodes


# When a move is made, it is processed here
def move(state, position):
    new_state = state[:]
    index = new_state.index(0)

    # If the movement was in the "up" direction, update the current state
    if position == 1:
        if index not in range(0, board_side):
            temp = new_state[index - board_side]
            new_state[index - board_side] = new_state[index]
            new_state[index] = temp
            return new_state
        else:
            return None

    # If the movement was in the "down" direction, update the current state
    if position == 2:
        if index not in range(board_len - board_side, board_len):
            temp = new_state[index + board_side]
            new_state[index + board_side] = new_state[index]
            new_state[index] = temp
            return new_state
        else:
            return None

    # If the movement was in the "left" direction, update the current state
    if position == 3:
        if index not in range(0, board_len, board_side):
            temp = new_state[index - 1]
            new_state[index - 1] = new_state[index]
            new_state[index] = temp
            return new_state
        else:
            return None

    # If the movement was in the "right" direction, update the current state
    if position == 4:
        if index not in range(board_side - 1, board_len, board_side):
            temp = new_state[index + 1]
            new_state[index + 1] = new_state[index]
            new_state[index] = temp
            return new_state
        else:
            return None


# Function to calculate for the first heuristic
def h1(state):
    h_state = sum(abs(b % board_side - g % board_side) + abs(b // board_side - g // board_side)
                  for b, g in ((state.index(i), goal_state.index(i)) for i in range(1, board_len)))
    return h_state


# Function to calculate for the second heuristic
def h2(state):
    count = 0
    for i in range(1, board_len):
        if state[i] != goal_state[i]:
            count += 1
    return count


# Function to backtrace the moves made when the game goal is reached
def backtrace():
    current_node = goal_node

    while initial_state != current_node.state:

        if current_node.move == 1:
            movement = 'Up'

        elif current_node.move == 2:
            movement = 'Down'

        elif current_node.move == 3:
            movement = 'Left'

        else:
            movement = 'Right'

        for i in range(len(current_node.state)):
            if current_node.state[i] == 0:
                current_node.state[i] = "*"

        # Print the game board for each move made in a successful route
        print("Movement:", movement)
        print(current_node.state[0], current_node.state[1], current_node.state[2])
        print(current_node.state[3], current_node.state[4], current_node.state[5])
        print(current_node.state[6], current_node.state[7], current_node.state[8], "\n")

        moves.insert(0, movement)
        current_node = current_node.parent

    return moves


# Function to display the initial input state, total number of moves, and number of states enqueued
def display():
    global moves

    moves = backtrace()

    for i in range(len(initial_state)):
        if initial_state[i] == 0:
            initial_state[i] = "*"

    # Print the initial input state of the board
    print(initial_state[0], initial_state[1], initial_state[2], "  (Initial input state)")
    print(initial_state[3], initial_state[4], initial_state[5])
    print(initial_state[6], initial_state[7], initial_state[8], "\n")
    print("Number of moves: " + str(len(moves)))
    print("Number of states enqueued: " + str(enqueued_states))


# Function to process the game board design
def read(configuration):
    global board_len, board_side

    # Split the game board parameter by whitespace
    data = configuration.split(" ")

    # If the game board is size 3x3 with unique values, then save the initial board state
    # If the game board is not size 3x3 with unique values, then exit the program
    if len(set(data)) == 9:
        for element in data:
            if element == "*":
                element = element.replace("*", "0")
            initial_state.append(int(element))
    else:
        print("Please enter a valid game board!")
        exit(0)

    board_len = len(initial_state)
    board_side = int(board_len ** 0.5)


def main():
    # Ensures that only two argument parameters are being passed
    # Save first argument parameter as the algorithm
    # Save second argument parameter as the board design
    if len(sys.argv) == 3:
        algorithm = sys.argv[1]
        board = sys.argv[2]

    # Call a function to process the board design
    read(board)

    # Read the first system argument parameter so we know which algorithm to run
    function = function_map[algorithm]

    # Run the desired algorithm, passing the initial game board as a parameter
    function(initial_state)

    display()


function_map = {
    'bfs': bfs,
    'ids': ids,
    'astar1': astar1,
    'astar2': astar2
}

if __name__ == '__main__':
    main()
