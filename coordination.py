# Author: Hugo Cruz Sanchez
# AMS 580 - RL project

import numpy as np
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, step_size, epsilon, grid_size, state):
        self.step_size = step_size  # learning rate
        self.epsilon = epsilon  # degree of exploration
        self.grid_size = grid_size  # number of cells in each dimension
        self.state = state  # a 4d tuple
        self.action_value = self.init_action_value()

    def init_action_value(self):
        aux = np.random.normal(
            size=(self.grid_size, self.grid_size,
                  self.grid_size, self.grid_size, 4))
        # action_value at terminal states is 0 for any action
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                aux[(i, j, i, j)] = 0
        return aux

    def choose_action(self, other_state=None):
        if np.random.uniform() > self.epsilon:
            if other_state is None:
                other_state = self.state
            action = self.choose_greedy(other_state)
        else:
            action = self.choose_free()
        return action

    def choose_greedy(self, other_state):
        # print("Greedy")
        return np.argmax(self.action_value[other_state])

    def choose_free(self):
        # print("Free")
        return np.random.choice(4)

    def update(self, action, other_state):
        other_action = self.choose_action(other_state)
        self.action_value[self.state][action] = \
            (1 - self.step_size) * self.action_value[self.state][action] +\
            self.step_size * (-1 + self.action_value[other_state][other_action])  # reward = -1 bcs it penalizes delay
        self.state = other_state
        return other_action

def go_through_episode(agent_1, action_1, agent_2, action_2, max_steps = 100000):
    def get_new_state():
        new_state = list(state)
        # if agent 1 tries to move out of the grid, it will stay in the same position
        if action_1 in (0, 1):
            new_state[0] = \
                np.minimum(grid_size - 1, np.maximum(0, state[0] + 2 * action_1 - 1))
        else:
            new_state[1] = \
                np.minimum(grid_size - 1, np.maximum(0, state[1] + 2 * action_1 - 5))
        # if agent 2 tries to move out of the grid, it will stay in the same position
        if action_2 in (0, 1):
            new_state[2] = \
                np.minimum(grid_size - 1, np.maximum(0, state[2] + 2 * action_1 - 1))
        else:
            new_state[3] = \
                np.minimum(grid_size - 1, np.maximum(0, state[3] + 2 * action_1 - 5))
        return tuple(new_state)

    grid_size = agent_1.grid_size  # common grid size
    state = agent_1.state  # common state: row_1, column_1, row_2, column_2; not terminal
    for step in range(max_steps):
        new_state = get_new_state()
        action_1 = agent_1.update(action_1, new_state)  # 0:up, 1:down, 2:left, 3:right
        action_2 = agent_2.update(action_2, new_state)  # 0:up, 1:down, 2:left, 3:right
        state = new_state
        agent_1.state = state
        agent_2.state = state
        # end of the episode if terminal state was found
        if state[:2] == state[2:]:
            break
    used_steps = step + 1
    return used_steps, agent_1, agent_2

def run_sarsa(agent_1, agent_2, tol = 1e-4, max_episodes = 100000, max_small_changes = 100, max_steps = 100000):
    grid_size = agent_1.grid_size  # common grid size
    join_old_action_value = np.concatenate((agent_1.action_value, agent_2.action_value))
    norm_old = np.amax(np.fabs(join_old_action_value))
    small_change = 0
    step_count = []
    episode_count = []
    for episode in range(max_episodes):
        # choose a non terminal state
        values = np.random.choice(grid_size ** 2, size=2, replace=False)
        row_1 = values[0] % grid_size; col_1 = values[0] // grid_size
        row_2 = values[1] % grid_size; col_2 = values[1] // grid_size
        state = (row_1, col_1, row_2, col_2)
        agent_1.state = state; agent_2.state = state
        action_1 = agent_1.choose_action()  # 0:up, 1:down, 2:left, 3:right
        action_2 = agent_2.choose_action()  # 0:up, 1:down, 2:left, 3:right
        used_steps, agent_1, agent_2 = \
            go_through_episode(agent_1, action_1, agent_2, action_2, max_steps)
        join_action_value = np.concatenate((agent_1.action_value, agent_2.action_value))
        norm_diff = np.amax(np.fabs(join_action_value - join_old_action_value))
        if episode % 100 == 0:
            print("Episode number %6d: they reached a terminal state in %6d steps "
                  "and the inf-norm of the difference between action value functions in 2 episodes was %.4f"
                  % (episode, used_steps, norm_diff))

            step_count.append(used_steps)
            episode_count.append(episode)

        if norm_diff < tol * norm_old:
            small_change += 1
        else:
            small_change = 0
        if small_change > max_small_changes:  # stop if max_small_changes subsequent small changes happened
            break
        join_old_action_value = join_action_value
        norm_old = np.amax(np.fabs(join_old_action_value))
        
    used_episodes = episode + 1
    return used_episodes, agent_1, agent_2, step_count, episode_count

# Main section
if __name__ == "__main__":
    np.random.seed(31416)
    # initializes two agents
    agent_1 = Agent(0.1, 0.01, 10, (0, 0, 9, 9))
    agent_2 = Agent(0.1, 0.01, 10, (0, 0, 9, 9))
    used_episodes, agent_1, agent_2, sc, ec = run_sarsa(agent_1, agent_2, max_episodes=10000)

    cumulative_step = 0
    cuml = []
    for i in sc:
        cumulative_step = cumulative_step + i
        cuml.append(cumulative_step)

    print("Used episodes:", used_episodes)
    fig = plt.figure(figsize=(50, 36))          
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ec, cuml, color='red', linestyle='-', marker='o')
    plt.show()