import numpy as np
import pygame
from pytorch_mlp import MLPRegression
import argparse
from console import FlappyBirdEnv
from collections import deque
import torch

STUDENT_ID = 'a1840406'
DEGREE = 'UG'  # or 'PG'


class MyAgent:
    def __init__(self, show_screen=False, load_model_path=None, mode=None):
        # do not modify these
        self.show_screen = show_screen
        if mode is None:
            self.mode = 'train'  # mode is either 'train' or 'eval', we will set the mode of your agent to eval mode
        else:
            self.mode = mode

        # modify these
        self.storage = deque(maxlen=10000)  # a data structure of your choice (D in the Algorithm 2)
        # A neural network MLP model which can be used as Q
        hidden_lay = (100, 250, 100, 50, 25)
        # hidden_lay = (200, 500, 100)
        self.network = MLPRegression(input_dim=9, output_dim=2, hidden_dim=hidden_lay, learning_rate=0.0007)
        # network2 has identical structure to network1, network2 is the Q_f
        self.network2 = MLPRegression(input_dim=9, output_dim=2, hidden_dim=hidden_lay, learning_rate=0.0007)
        # initialise Q_f's parameter by Q's, here is an example
        MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)

        self.epsilon = 1  # probability ε in Algorithm 2
        self.n = 64  # the number of samples you'd want to draw from the storage each time
        self.discount_factor = 0.99  # γ in Algorithm 2
        self.previous_score = 0

        # do not modify this
        if load_model_path:
            self.load_model(load_model_path)

    def choose_action(self, state: dict, action_table: dict) -> int:
        """
            state: input state representation (the state dictionary from the game environment)
            action_table: the action code dictionary
        """
        # following pseudocode to implement this function
        a_t = None
        
        state_representation = self.BUILD_STATE(state)
        
        if self.mode == 'train':
            # With probability epsilon, select a random action
            if np.random.rand() < self.epsilon:
                a_t = np.random.choice(list(action_table.values()))
                # if action is quit_game, choose another action
                while a_t == action_table["quit_game"]:
                    a_t = np.random.choice(list(action_table.values()))
            else:
                # Predict Q-values for both actions and choose the best action
                q_values = self.network(state_representation)
                a_t = int(np.argmax(q_values.detach().numpy()))
            
            # Store partial transition in replay memory regardless of how the action was chosen.
            self.storage.append((state_representation, a_t))
        elif self.mode == 'eval':
            q_values = self.network(state_representation)
            a_t = int(np.argmax(q_values.detach().numpy()))
            
        return a_t

    def receive_after_action_observation(self, state: dict, action_table: dict) -> None:
        '''
        state: state of the game at t+1 after an action is taken at t
        action_table: the action code dictionary
        '''
        if self.mode == 'train':
            current_state_features = self.BUILD_STATE(state)
            current_reward = self.REWARD(state)
            
            # update the last stored transition with the current state and reward
            prev_state_features, prev_action = self.storage[-1][0], self.storage[-1][1]
            self.storage[-1] = (prev_state_features, prev_action, current_reward, current_state_features)
            
            # train if have enough samples
            if len(self.storage) >= self.n:  # try making it longer to see if it helps (tried x2 did not help)
                # Filter out quit_game actions
                valid_indices = list(range(len(self.storage)))
                if len(valid_indices) == 0:
                    return

                # Sample batch of transitions
                batch_size = min(self.n, len(valid_indices))
                sampled_indices = np.random.choice(valid_indices, batch_size, replace=False)

                state_batch = []
                target_batch = []
                weight_batch = []
                
                for idx in sampled_indices:
                    state_t, action_t, reward_t, next_state_t = self.storage[idx]
                    
                    # Get next state Q-values from target network
                    if (next_state_t is not None) and (not state['done']):
                        next_q_values = self.network2(next_state_t.unsqueeze(0))
                        next_q_array = next_q_values.detach().numpy()[0]
                        max_next_q = np.max(next_q_array)
                    else:
                        max_next_q = 0.0
                        
                    # Compute TD target
                    td_target = reward_t + self.discount_factor * max_next_q
                    
                    # Get current Q-values and create target vector
                    current_q_values = self.network(state_t.unsqueeze(0)).detach().numpy()[0]
                    target_q_values = current_q_values.copy()
                    
                    # Convert action to index if needed
                    if not isinstance(action_t, int):
                        action_idx = list(action_table.values()).index(action_t)
                    else:
                        action_idx = action_t
                    
                    # Update target for chosen action
                    target_q_values[action_idx] = td_target
                    
                    # Add to batch
                    state_batch.append(state_t.numpy())
                    target_batch.append(target_q_values)
                    weight_batch.append(np.ones_like(target_q_values, dtype=np.float32))
                
                # Convert batches to numpy arrays
                state_batch_np = np.array(state_batch, dtype=np.float32)
                target_batch_np = np.array(target_batch, dtype=np.float32)
                weight_batch_np = np.array(weight_batch, dtype=np.float32)
                
                # Update Q-network
                self.network.fit_step(state_batch_np, target_batch_np, weight_batch_np)
                
                # Decay exploration rate
                self.epsilon = max(0.1, self.epsilon * 0.999)
                    

    def save_model(self, path: str = 'my_model.ckpt'):
        """
        Save the MLP model. Unless you decide to implement the MLP model yourself, do not modify this function.

        Args:
            path: the full path to save the model weights, ending with the file name and extension

        Returns:

        """
        self.network.save_model(path=path)

    def load_model(self, path: str = 'my_model.ckpt'):
        """
        Load the MLP model weights.  Unless you decide to implement the MLP model yourself, do not modify this function.
        Args:
            path: the full path to load the model weights, ending with the file name and extension

        Returns:

        """
        self.network.load_model(path=path)

    @staticmethod
    def update_network_model(net_to_update: MLPRegression, net_as_source: MLPRegression):
        """
        Update one MLP model's model parameter by the parameter of another MLP model.
        Args:
            net_to_update: the MLP to be updated
            net_as_source: the MLP to supply the model parameters

        Returns:
            None
        """
        net_to_update.load_state_dict(net_as_source.state_dict())
        
    def ONEHOT(self, action: int, action_table: dict) -> torch.Tensor:
        """
            action: action code
            action_table: the action code dictionary        
        """
        action_onehot = np.zeros(len(action_table))
        action_onehot[action] = 1
        return torch.tensor(action_onehot, dtype=torch.float32)
        
    def BUILD_STATE(self, state: dict) -> torch.Tensor:
        # screen dimensions
        screen_width = state['screen_width']
        screen_height = state['screen_height']
        
        # normalize bird parameters
        normalized_bird_x = state['bird_x'] / screen_width
        normalized_bird_y = state['bird_y'] / screen_height
        normalized_bird_width = state['bird_width'] / screen_width
        normalized_bird_height = state['bird_height'] / screen_height
        normalized_bird_velocity = state['bird_velocity'] / 10.0  # assuming maximum expected velocity is ~10
        
        # Find the first pipe that is ahead of the bird
        upcoming_pipe = None
        for pipe in state['pipes']:
            if pipe['x'] + pipe['width'] > state['bird_x']:
                upcoming_pipe = pipe
                break
        
        if upcoming_pipe:
            normalized_pipe_distance = (upcoming_pipe['x'] - state['bird_x']) / screen_width
            normalized_pipe_top = (upcoming_pipe['top'] - state['bird_y']) / screen_height
            normalized_pipe_bottom = (upcoming_pipe['bottom'] - state['bird_y']) / screen_height
            normalized_pipe_width = upcoming_pipe['width'] / screen_width
        else:
            # if no pipe is found, use defaults based on screen boundaries
            normalized_pipe_distance = (screen_width - state['bird_x']) / screen_width
            normalized_pipe_top = - state['bird_y'] / screen_height
            normalized_pipe_bottom = (screen_height - state['bird_y']) / screen_height
            normalized_pipe_width = 0.0

        # Build the final normalized state tensor
        normalized_state_tensor = torch.tensor([
            normalized_bird_x,
            normalized_bird_y,
            normalized_bird_width,
            normalized_bird_height,
            normalized_bird_velocity,
            normalized_pipe_distance,
            normalized_pipe_top,
            normalized_pipe_bottom,
            normalized_pipe_width,
        ], dtype=torch.float32)
        
        return normalized_state_tensor
    
    # def BUILD_STATE(self, state: dict) -> torch.Tensor:
    #     # screen dimensions
    #     screen_width = state['screen_width']
    #     screen_height = state['screen_height']
        
    #     # normalize bird velocity
    #     normalized_bird_velocity = state['bird_velocity'] / 10.0  
        
    #     # Find the first pipe that is ahead of the bird
    #     upcoming_pipe = None
    #     for pipe in state['pipes']:
    #         if pipe['x'] + pipe['width'] > state['bird_x']:
    #             upcoming_pipe = pipe
    #             break
        
    #     if upcoming_pipe:
    #         # horizontal distance to the pipe
    #         normalized_pipe_distance = (upcoming_pipe['x'] - state['bird_x']) / screen_width
            
    #         # Compute the center of the gap.
    #         gap_center = (upcoming_pipe['top'] + upcoming_pipe['bottom']) / 2.0
    #         # vertical offset from gap centered
    #         normalized_vertical_offset = (state['bird_y'] - gap_center) / screen_height
    #     else:
    #         # if no pipe is found, use defaults based on screen boundaries
    #         normalized_pipe_distance = (screen_width - state['bird_x']) / screen_width
    #         normalized_vertical_offset = - state['bird_y'] / screen_height

    #     # Build the final normalized state tensor
    #     normalized_state_tensor = torch.tensor([
    #         normalized_vertical_offset,
    #         normalized_pipe_distance,
    #         normalized_bird_velocity,
    #     ], dtype=torch.float32)
        
    #     return normalized_state_tensor

    # def REWARD(self, state: dict) -> float:
    #     """
    #     +0.1 for staying alive
    #     +1.0 for passing a pipe
    #     + adaptive reward for barely passing a pipe
    #     -1.0 for dying
    #     - offset for the gap center
    #     """
    #     # if dead return -1
    #     if state['done']:
    #         return -1.0

    #     # reward for staying alive.
    #     reward = 0.1

    #     # Find the first pipe that the bird has not yet passed.
    #     next_pipe = None
    #     for pipe in state['pipes']:
    #         if pipe['x'] + pipe['width'] > state['bird_x']:
    #             next_pipe = pipe
    #             break

    #     if next_pipe is not None:
    #         pipe_right_edge = next_pipe['x'] + next_pipe['width']
    #         bird_right_edge = state['bird_x'] + state['bird_width']
            
    #         # if pass, reward +1
    #         if pipe_right_edge < state['bird_x']:
    #             reward += 1.0
    #         # adaptive reward for barely passing
    #         elif abs(pipe_right_edge - bird_right_edge) < 5:
    #             # The closer the bird is to the pipe, the higher the reward, up to a maximum of 0.5
    #             reward += 0.5 * (5 - abs(pipe_right_edge - bird_right_edge)) / 5

    #         # Compute the center of the gap.
    #         gap_center = (next_pipe['top'] + next_pipe['bottom']) / 2.0
    #         # reward based on the distance to the gap center
    #         reward -= abs(state['bird_y'] - gap_center) / state['screen_height']

    #     return reward
    
    def REWARD(self, state: dict) -> float:
        """
        Reward function:
        -1.0 for dying
        +0.1 for each frame survived
        +1.0 for passing a pipe
        - small penalty for being far from the gap center
        """
        # Check if the bird is dead
        if state['done']:
            return -1.0

        # Reward for staying alive
        reward = 0.1

        # Check if the bird has passed a pipe
        # if pass, reward +1
        if state['score'] > self.previous_score:
            reward += (state['score'] - self.previous_score) * 1.0
            self.previous_score = state['score']

        # Find the first pipe that the bird has not yet passed
        next_pipe = None
        for pipe in state['pipes']:
            if pipe['x'] + pipe['width'] > state['bird_x']:
                next_pipe = pipe
                break

        if next_pipe is not None:
            # Compute the center of the gap
            gap_center = (next_pipe['top'] + next_pipe['bottom']) / 2.0
            # Penalize based on the distance to the gap center
            distance_to_gap_center = abs(state['bird_y'] - gap_center)
            max_penalty = 0.5
            penalty = max_penalty * (distance_to_gap_center / state['screen_height'])
            reward -= penalty

        return reward


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=7)

    args = parser.parse_args()

    # bare-bone code to train your agent (you may extend this part as well, we won't run your agent training code)
    # best_score = 0
    # best_mileage = 0
    # best_score_episode = 0
    # total_episodes = 10000

    # # Initialize the agent once to be used across all levels
    # agent = MyAgent(show_screen=False)

    # # Define the training schedule
    # training_schedule = [
    #     # (5, 2000),  # Train 2000 episodes on level 5
    #     (6, 3000),  # Train 3000 episodes on level 6
    #     (7, 5000)   # Train 5000 episodes on level 7
    # ]

    # current_episode = 0

    # for level, episodes in training_schedule:
    #     env = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=level, game_length=50)

    #     # Initialize best score and mileage for each level
    #     best_score = 0
    #     best_mileage = 0

    #     for episode in range(episodes):
    #         env.play(player=agent)

    #         print("--Episode {}--".format(current_episode))
    #         print(env.score)
    #         print(env.mileage)
    #         print("Current level best score: ", best_score)
    #         print("Current level best mileage: ", best_mileage)
    #         print("Current level best score episode: ", best_score_episode)

    #         # store the best model based on your judgement for each level
    #         if env.score > best_score or env.mileage > best_mileage:
    #             best_mileage = env.mileage
    #             best_score = env.score
    #             best_score_episode = current_episode
    #             print('NEW BEST')
    #             # save the model
    #             agent.save_model(path='my_model.ckpt')

    #         # rely on deque to clear the memory
    #         if current_episode % 1000 == 0 and len(agent.storage) > agent.n:
    #             purge = len(agent.storage) // 3
    #             for _ in range(purge):
    #                 agent.storage.popleft()
    #             print(f"Cleared {purge} oldest records")

    #         # you'd want to update the fixed Q-target network (Q_f) with Q's model parameter after one or a few episodes
    #         if current_episode % 100 == 0:
    #             agent.update_network_model(net_to_update=agent.network2, net_as_source=agent.network)

    #         current_episode += 1

    # print("Max score: ", best_score)
    # print("Max mileage: ", best_mileage)
    # print("Best score episode: ", best_score_episode)

    # the below resembles how we evaluate your agent
    env2 = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level)
    agent2 = MyAgent(show_screen=False, load_model_path='my_model.ckpt', mode='eval')
    agent2.network.global_prune_neurons(amount=0.1)
    agent2.save_model(path='my_model_pruned.ckpt')

    episodes = 30
    scores = list()
    for episode in range(episodes):
        env2.play(player=agent2)
        scores.append(env2.score)

    print("Max score: ", np.max(scores))
    print("Mean score: ", np.mean(scores))
    print("Median score: ", np.median(scores))