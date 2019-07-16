import gym
import numpy as np
import random


# Print all states
directions = ['LEFT', 'DOWN', 'RIGHT', 'UP']
Q = {}


def printQ():
    lastKey = 0
    first = True
    print("Action:\t\t   " + directions[0] + "\t\t\t   " +
          directions[1] + "\t\t\t   " + directions[2] + "\t\t   " + directions[3])
    for key, value in Q.items():
        if lastKey != key[0]:
            print()
            lastKey = key[0]
            first = True
        if first:
            print("State: " + str(key[0]), end='\t')
            first = False
        print(f'{value:9.4f}', end='\t||\t')
    print()
    env.render()


def initializeQ():
    global Q
    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            # it is more probable to go right or down
            if action > 0 and action < 3:
                Q[(state, action)] = random.uniform(0.55, 0.60)
            else:
                Q[(state, action)] = random.uniform(0.45, 0.50)


def moveIt(state):  # Softmax
    stateActionPairs = [(state, action)
                        for action in range(env.action_space.n)]
    probabilities = [Q[saPair] for saPair in stateActionPairs]
    normalizedP = [0] * env.action_space.n

    pSum = sum([3**x for x in probabilities])
    for idx, p in enumerate(probabilities):
        p = 3**p / pSum
        if idx > 0:
            normalizedP[idx] = p + normalizedP[idx-1]
        else:
            normalizedP[idx] = p

    choice = random.uniform(0, 1)
    counter = 0
    while choice > normalizedP[counter]:
        counter += 1

    return counter


def myReward(prevState, newState):
    if newState == BOARD_SIZE * BOARD_SIZE - 1:
        return 100

    if env.desc[newState // BOARD_SIZE][newState % BOARD_SIZE] == b'H':
        return -50

    rowDiff = newState // BOARD_SIZE - prevState // BOARD_SIZE
    colDiff = newState % BOARD_SIZE - prevState % BOARD_SIZE

    if rowDiff > 0 or colDiff > 0:
        return 2

    return -1


def learnQ(DECAY_RATE=0.9, LEARNING_RATE=0.3, MAX_ITERATION=10000, LR_SCHEDULE=1000, LR_DECAY=0.1):
    initializeQ()

    for i in range(MAX_ITERATION):
        state, reward, done = 0, 0, False
        env.reset()
        if i != 0 and i % LR_SCHEDULE == 0:
            LEARNING_RATE = LEARNING_RATE * LR_DECAY
        while not done:
            move = moveIt(state)
            newState, _, done, _ = env.step(move)
            newReward = myReward(state, newState)
            r = newReward - reward

            maxQNewState = max([Q[(newState, a)]
                                for a in range(env.action_space.n)])
            difference = LEARNING_RATE * \
                (r + DECAY_RATE * maxQNewState - Q[(state, move)])
            Q[(state, move)] = Q[(state, move)] + difference

            reward = newReward
            state = newState


def runQ(silent=True):
    env.reset()

    state, done = 0, False
    if not silent:
        env.render()
    while not done:
        state, reward, done, _ = env.step(moveIt(state))
        if not silent:
            env.render()

    return reward


def winRateCalc(kind, runs=10000):
    rewards = 0
    for _ in range(runs):
        rewards += runQ()
    x = (rewards / runs) * 100
    print("Q-learning win ratio for {} : {}%".format(kind,
                                                     f'{x:4.2f}'))


def showOff(howMuch):
    for _ in range(howMuch):
        learnQ(MAX_ITERATION=10000, LR_SCHEDULE=1000, LR_DECAY=0.1)
        winRateCalc(kind="{}x{}".format(BOARD_SIZE, BOARD_SIZE))

# --------------------------------------------------------------------------
# random approach


def simulateOne(env):
    done = False
    while not done:
        # randomly step towards Down or Right
        _, reward, done, _ = env.step(random.randint(1, 2))

    return reward  # 1 if won else 0


def simulate(kind, nIteration):
    env = gym.make("FrozenLake-v0", map_name=kind)
    wins = 0
    for _ in range(nIteration):
        env.reset()
        wins += simulateOne(env)

    return (wins / nIteration) * 100


n_iter = 10000
for kind in ['4x4', '8x8']:
    print("Random win ratio for {} : {}%".format(kind, simulate(kind, n_iter)))

# --------------------------------------------------------------------------

BOARD_SIZE = 4

# Make board
env = gym.make("FrozenLake-v0",
               map_name="{}x{}".format(BOARD_SIZE, BOARD_SIZE))

showOff(10)

# printQ()

# --------------------------------------------------------------------------

BOARD_SIZE = 8

# Make board
env = gym.make("FrozenLake-v0",
               map_name="{}x{}".format(BOARD_SIZE, BOARD_SIZE))

showOff(10)

# printQ()

# --------------------------------------------------------------------------
