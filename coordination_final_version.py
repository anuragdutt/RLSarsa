# Author: Hugo Cruz Sanchez
# AMS 580 - RL project

# Page 130, Barto and Sutton
#
# SARSA (on-policy TD control) for estimating q, the optimal action value function
#
# Algorithm parameters: step_size in (0, 1], small epsilon > 0
# Initialize q(s, a), for all s in S+, a in A(s), arbitrarily except that q(terminal , ·) = 0
#
# Loop for each episode:    (step 1)
#   Initialize s    (step 1.1)
#   Choose a from s using policy derived from q (e.g., epsilon-greedy)  (step 1.2)
#   Loop for each step of episode:  (step 1.3)
#       Take action a, observe r, s'    (step 1.3.1)
#       Choose a' from s' using policy derived from q (e.g., epsilon-greedy)    (step 1.3.2)
#       q(s, a) <- q(s, a) + step_size * (r + q(s', a') - q(s, a))  (step 1.3.3)
#       s <- s';  a <- a'   (step 1.3.4)
#   until S is terminal (step 1.4)

import matplotlib.pyplot as mp
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib import colors
import numpy as np
from time import perf_counter


class Agent:
    def __init__(self, step_size, epsilon, grid_size, state):
        # Initialize agents
        self.step_size = step_size  # learning rate
        self.epsilon = epsilon  # degree of exploration
        self.grid_size = grid_size  # number of cells in each dimension
        self.state = state  # a 4d tuple
        # action value function Q is initialized with 0s
        self.action_value = np.zeros(shape=(grid_size, grid_size, grid_size, grid_size, 4))

    def choose_action(self, other_state=None):  # epsilon-greedy 
        # choose an action for the current state or other_state
        if np.random.uniform() > self.epsilon:  # greedy choice - exploitation
            if other_state is None: # other_state is the current state
                other_state = self.state
            action = np.argmax(self.action_value[other_state])  # greedy choice among 0 to 3
        else:  # random choice - exploration
            action = np.random.choice(4)  # random choice among 0 to 3
        return action

    def update(self, action, other_state):
        other_action = self.choose_action(other_state)  # (step 1.3.2)
        # reward = -1 bcs it penalizes delay
        reward = -1  # from (step 1.3.1)
        # (step 1.3.3)
        self.action_value[self.state][action] = \
            (1 - self.step_size) * self.action_value[self.state][action] + self.step_size *\
            (reward + self.action_value[other_state][other_action])
        self.state = other_state  # (step 1.3.4)
        return other_action  # (step 1.3.4)

def get_new_state(grid_size, state, action_1, action_2):
    new_state = list(state)
    
    # if agent 1 tries to move out of the grid, it will stay in the same position
    if action_1 in (0, 1):  # 0:up, 1:down
        new_state[0] = \
            np.minimum(grid_size - 1, np.maximum(0, state[0] + 2 * action_1 - 1))
    else:  # 2:left, 3:right
        new_state[1] = \
            np.minimum(grid_size - 1, np.maximum(0, state[1] + 2 * action_1 - 5))
        
    # if agent 2 tries to move out of the grid, it will stay in the same position
    if action_2 in (0, 1):  # 0:up, 1:down
        new_state[2] = \
            np.minimum(grid_size - 1, np.maximum(0, state[2] + 2 * action_2 - 1))
    else:  # 2:left, 3:right
        new_state[3] = \
            np.minimum(grid_size - 1, np.maximum(0, state[3] + 2 * action_2 - 5))
        
    return tuple(new_state)

def go_through_episode(agent_1, action_1, agent_2, action_2, max_steps = 100000):
    # this routine is (step 1.3)
    grid_size = agent_1.grid_size  # common grid size
    state = agent_1.state  # common state: row_1, column_1, row_2, column_2; not terminal
    for step in range(max_steps):  # (step 1.3)
        # given agents actions, observe the next state s' - (step 1.3.1) without r
        new_state = get_new_state(grid_size, state, action_1, action_2)
        # observe r from (step 1.3.1) and execute (step 1.3.2), (step 1.3.3), and (step 1.3.4)
        action_1 = agent_1.update(action_1, new_state)  # 0:up, 1:down, 2:left, 3:right
        action_2 = agent_2.update(action_2, new_state)  # 0:up, 1:down, 2:left, 3:right
        # local update of the variable state with the new state for agents (updated in (step 1.3.4))
        state = new_state
        # end of the episode if terminal state was found - (step 1.4)
        if state[:2] == state[2:]:
            break
    used_steps = step + 1   # extra info - not important/not essential
    return used_steps, agent_1, agent_2

def run_sarsa(agent_1, agent_2, tol = 1e-4, max_episodes = 100000, max_small_changes = 100, max_steps = 100000):
    global conv_metrics  # global variable that stores convergence metrics

    # this routine is (step 1)
    grid_size = agent_1.grid_size  # common grid size

    used_steps_1000 = np.zeros(1000)  # store used steps in episodes - not important/not essential
    idx = 0  # count episodes - not important/not essential
    # used to measure convergence - not important/not essential
    join_old_action_value = \
        np.concatenate((agent_1.action_value, agent_2.action_value))
    # used to measure convergence - not important/not essential
    norm_old = \
        np.amax(np.fabs(join_old_action_value))
    # used to measure convergence - not important/not essential
    small_change = 0

    for episode in range(max_episodes):  # (step 1)
        # choose a non terminal state, randomly - (step 1.1)
        values = np.random.choice(grid_size ** 2, size=2, replace=False)
        row_1 = values[0] % grid_size; col_1 = values[0] // grid_size
        row_2 = values[1] % grid_size; col_2 = values[1] // grid_size
        state = (row_1, col_1, row_2, col_2)
        agent_1.state = state; agent_2.state = state  # assign the state to agents

        # choose actions from state using policy derived from action value function - (step 1.2)
        action_1 = agent_1.choose_action()  # 0:up, 1:down, 2:left, 3:right
        action_2 = agent_2.choose_action()  # 0:up, 1:down, 2:left, 3:right

        # Loop for each step of episode:  (step 1.3)
        used_steps, agent_1, agent_2 = \
            go_through_episode(agent_1, action_1, agent_2, action_2, max_steps)

        # used to measure convergence - not important/not essential
        used_steps_1000[idx] = used_steps
        idx += 1

        # info about convergence - not important/not essential
        if idx % 1000 == 0:
            # used to measure convergence - not important/not essential
            join_action_value = \
                np.concatenate((agent_1.action_value, agent_2.action_value))
            # used to measure convergence - not important/not essential
            norm_diff = \
                np.amax(np.fabs(join_action_value - join_old_action_value))
            # used to measure convergence - not important/not essential
            if norm_diff < tol * norm_old:
                small_change += 1
            else:
                small_change = 0
            # used to measure convergence - not important/not essential
            join_old_action_value = join_action_value
            # used to measure convergence - not important/not essential
            norm_old = np.amax(np.fabs(join_old_action_value))

            # store convergence metrics
            m = np.mean(used_steps_1000)
            sd = np.std(used_steps_1000)
            lb = np.percentile(used_steps_1000, 0.025)
            ub = np.percentile(used_steps_1000, 0.975)
            conv_metrics.append(
                [episode + 1, lb, m, ub, norm_diff]
            )

            print("Episodes: %6d; Terminal state after - mean (sd): %.2f (%.2f) steps; "
                  "Inf-norm of the difference between action value functions (lag=1000): %.4f"
                  % (episode + 1, m, sd, norm_diff))

            idx = 0

        # used to measure convergence - not important/not essential
        if small_change > max_small_changes:  # stop if max_small_changes subsequent small changes happened
            break

    used_episodes = episode + 1  # used to measure convergence - not important/not essential
    return used_episodes, agent_1, agent_2  # return agents with optimal action value functions

def simulate(state, agent_1, agent_2, file_path):
    # this is just a routine that generates gif files
    # from simulations, i.e. starting from a state, the agents follow
    # their action value functions (greedy) until they reach a terminal state
    def animate(i):
        state = stages[i]
        grid = np.zeros(shape=(grid_size, grid_size))
        grid[state[:2]] = 1; grid[state[2:]] = 2
        ax.imshow(grid, cmap=cmap, norm=norm)
        ax.set_title("Step: %2d" % i, fontsize=20)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_xticks(np.arange(-0.5, grid_size - 0.5, 1))
        ax.set_yticks(np.arange(-0.5, grid_size - 0.5, 1))
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.grid(True)
        # ax.set_axis_off()

    agent_1.epsilon = 0; agent_2.epsilon = 0  # just greedy
    grid_size = agent_1.grid_size  # common grid size
    stages = []
    for step in range(30):
        stages.append(state)
        action_1 = agent_1.choose_action(state)  # 0:up, 1:down, 2:left, 3:right
        action_2 = agent_2.choose_action(state)  # 0:up, 1:down, 2:left, 3:right
        if state[:2] == state[2:]:
            break
        new_state = get_new_state(grid_size, state, action_1, action_2)
        state = new_state
    fig, ax = mp.subplots(figsize=(5, 5))
    cmap = colors.ListedColormap(["white", "red", "blue"])
    boundaries = [-0.5, 0.5, 1.5, 2.5]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    animation = FuncAnimation(fig, animate, frames=np.arange(step + 1), interval=50, repeat=False)
    animation.save(file_path, writer='imagemagick', fps=3)
    mp.show()
    mp.close()
    return None

# Main section
if __name__ == "__main__":
    # store convergence metrics
    conv_metrics = []
    # initializes two agents
    agent_1 = Agent(0.1, 0.01, 10, (0, 0, 9, 9))
    agent_2 = Agent(0.1, 0.01, 10, (0, 0, 9, 9))
    starting_time = perf_counter()
    np.random.seed(31416)
    used_episodes, agent_1, agent_2 = run_sarsa(agent_1, agent_2, max_episodes=1000000, max_steps=10000)
    ending_time = perf_counter()
    print("Used episodes: %6d; Execution time: %.0f" % (used_episodes, ending_time - starting_time))

    # Convergence graphs
    cm = np.array(conv_metrics)
    fig, (ax1, ax2) = mp.subplots(1, 2)
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Number of steps')
    ax1.plot(cm[:, 0], cm[:, 1], label="Percentile(2.5%)")
    ax1.plot(cm[:, 0], cm[:, 2], label="Mean")
    ax1.plot(cm[:, 0], cm[:, 3], label="Percentile(97.5%)")
    ax1.set_title("Number of steps to terminal state")
    ax1.legend()
    ax2.plot(cm[:, 0], cm[:, 4])
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Max. diff.')
    ax2.set_title("Inf-norm of the difference between join action value functions (lag=1000)")
    fig.suptitle("Convergence metrics")
    mp.savefig("Convergence metrics", dpi=600)
    mp.show()
    mp.close()

    simulate((0, 0, 9, 9), agent_1, agent_2, "animation (0, 0, 9, 9).gif")

    simulate((9, 0, 9, 9), agent_1, agent_2, "animation (9, 0, 9, 9).gif")

    simulate((0, 9, 9, 9), agent_1, agent_2, "animation (0, 9, 9, 9).gif")

    simulate((0, 5, 9, 5), agent_1, agent_2, "animation (0, 5, 9, 5).gif")

    simulate((5, 0, 5, 9), agent_1, agent_2, "animation (5, 0, 5, 9).gif")






