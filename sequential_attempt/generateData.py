import numpy as np

from cube import Cube, getAllMoves, getInverseMove, getRandomMove    
from bad.csvdata import saveData

def addAllStatesWithinThreeMoves(sequential=False):
        cube = Cube()

        labels = []
        states = []

        #initial solved state
        state0 = cube.getStateArray()
        states.append(state0)
        labels.append(0) # the "no move"

        #first moves
        for i in getAllMoves():
            cube.reset()

            cube.move(i)
            state1 = cube.getStateArray()
            inverseMove1 = getInverseMove(i)

            if not sequential:
                states.append(state1)
                labels.append(inverseMove1)

            for j in getAllMoves(i):
                cube.move(j)
                state2 = cube.getStateArray()
                inverseMove2 = getInverseMove(j)

                if not sequential:
                    states.append(state2)
                    labels.append(inverseMove2)

                for k in getAllMoves(j):
                    cube.move(k)
                    state3 = cube.getStateArray()
                    inverseMove3 = getInverseMove(k)
                    if not sequential:
                        states.append(state3)
                        labels.append(inverseMove3)
                    else:
                        # all 3 moves + solved
                        states.append([state0, state1, state2, state3])
                        labels.append(0)
                        # all 3 moves upto solved
                        states.append([state1, state2, state3])
                        labels.append(inverseMove1)
                        # first 2 moves + solved
                        states.append([state0, state1, state2])
                        labels.append(0)
                        # first 2 moves upto solved
                        states.append([state1, state2])
                        labels.append(inverseMove1)
                        # last 2 moves
                        states.append([state2, state3])
                        labels.append(inverseMove2)
                        # first move + solved
                        states.append([state0, state1])
                        labels.append(0)
                        # first move upto solved
                        states.append([state1])
                        labels.append(inverseMove1)
                        # middle move
                        states.append([state2])
                        labels.append(inverseMove2)
                        # last movea
                        states.append([state3])
                        labels.append(inverseMove3)

                        # # pad recent state with arrays of 54 -1's until theres 10 states in each
                        # while len(states[-1]) < 10:
                        #     states[-1].append(list(np.full(54, -1)))

        # print(len(states))
        # print(len(labels))

        return labels, states



def getRandomData(startWithXMoves=0, maxMoves=17, numDataPoints=30000, sequential=False):
    dataPointsToAdd = numDataPoints
    print("Generating data...")
    labels = []
    states = []
    
    # Maintain a set of states as tuples for efficient O(1) membership tests
    seen_states = set()

    while dataPointsToAdd > 0:
        if dataPointsToAdd % 500 == 0:
            print("...", dataPointsToAdd, "left")
        cube = Cube()

        # Optionally scramble the cube with startWithXMoves
        for _ in range(startWithXMoves):
            cube.move(getRandomMove())

        moves = []
        sequence = []
        prevMove = None
        for j in range(maxMoves):
            move = getRandomMove(prevMove)
            cube.move(move)
            prevMove = move
            moves.append(move)

            # Get the cube state as a list (needed by saveData)
            state_list = cube.getStateArray()
            sequence.append(state_list)

        # For sequential mode
        randomCutoff = np.random.randint(1, maxMoves)
        if sequential:
            # Append the slice of states and a single label
            states.append(sequence[randomCutoff:])
            labels.append(getInverseMove(moves[randomCutoff]))
        else:
            # For non-sequential mode, record each unique state
            for idx, state_list in enumerate(sequence):
                # Convert to tuple for set membership checking
                state_tuple = tuple(state_list)
                if state_tuple not in seen_states:
                    # Add to states as a list (for compatibility with saveData/loadData)
                    states.append(state_list)
                    labels.append(getInverseMove(moves[idx]))
                    seen_states.add(state_tuple)
                    dataPointsToAdd -= 1
                    if dataPointsToAdd == 0:
                        break

    return labels, states

def generateData():
    labels = []
    states = []

    # If you have other data sources, you can combine them here
    l, s = getRandomData()
    labels.extend(l)
    states.extend(s)

    return labels, states



labels, states = generateData()
saveData(labels, states)

