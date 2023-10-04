import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import random

torch.manual_seed(0)
#-------------En pruebas-----------------------
class MinMax_alfa_beta:
    """Implementacion del algoritmo de busqueda minmax con poda alpha-beta para el juego de Conecta 6"""
    def __init__(self, game, args):
        self.game = game
        self.args = args
    
    def search(self, state):
        """Busca la mejor accion para el estado actual"""
        _, action = self.max_value(state, -np.inf, np.inf, self.args['depth'])
        return action
    
    def max_value(self, state, alpha, beta, depth):
        """Calcula el valor maximo para el jugador max"""
        if depth == 0:
            return self.game.get_heuristic_value(state), None
        
        value = -np.inf
        best_action = None
        for action in np.where(self.game.get_valid_moves(state) == 1)[0]:
            child_state = self.game.get_next_state(state, action, 1)
            child_value, _ = self.min_value(child_state, alpha, beta, depth - 1)
            if child_value > value:
                value = child_value
                best_action = action
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, best_action
    
    def min_value(self, state, alpha, beta, depth):
        """Calcula el valor minimo para el jugador min"""
        if depth == 0:
            return self.game.get_heuristic_value(state), None
        
        value = np.inf
        best_action = None
        for action in np.where(self.game.get_valid_moves(state) == 1)[0]:
            child_state = self.game.get_next_state(state, action, -1)
            child_value, _ = self.max_value(child_state, alpha, beta, depth - 1)
            if child_value < value:
                value = child_value
                best_action = action
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value, best_action
    
    def get_action_probs(self, state):
        """Calcula la distribucion de probabilidad de las acciones"""
        action_probs = np.zeros(self.game.action_size)
        action_probs[self.search(state)] = 1
        return action_probs
    
    def get_action(self, state):
        """Calcula la mejor accion"""
        return self.search(state)
    
    
#--------------------------------------------------------------  



class MCTSNode:
    """
    Node of MCTS tree
    """

    def __init__(self, game, args, state, parent=None, action_taken=None):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        
        self.children = []
        self.expandable_moves = game.get_valid_moves(state)
        
        self.visit_count = 0
        self.value_sum = 0
        
    def is_fully_expanded(self):
        """Check if all moves are expanded"""
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0
    
    def select(self):
        """Select the best child node"""
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        """Calculate UCB value"""
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)
    
    def expand(self):
        """Expand a random unexpanded move"""
        action = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[action] = 0
        
        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, action, 1)
        child_state = self.game.change_perspective(child_state, player=-1)
        
        child = MCTSNode(self.game, self.args, child_state, self, action)
        self.children.append(child)
        return child
    
    def simulate(self):
        value, is_terminal = self.game.get_value_and_terminated(self.state, self.action_taken)
        value = self.game.get_opponent_value(value)
        
        if is_terminal:
            return value
        
        rollout_state = self.state.copy()
        rollout_player = 1
        while True:
            valid_moves = self.game.get_valid_moves(rollout_state)
            action = np.random.choice(np.where(valid_moves == 1)[0])
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
            value, is_terminal = self.game.get_value_and_terminated(rollout_state, action)
            if is_terminal:
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value)
                return value    
            
            rollout_player = self.game.get_opponent(rollout_player)
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  


class MCTS:
    """
    Monte Carlo Tree Search Algorithm (classic)
    """

    def __init__(self, game, args):
        self.game = game
        self.args = args
        
    def search(self, state):
        root = MCTSNode(self.game, self.args, state)
        
        for search in range(self.args['num_searches']):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                node = node.expand()
                value = node.simulate()
                
            node.backpropagate(value)    
            
            
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
    



# ALPHAMCTS
#--------------------------------------------------------------

import torch.nn as nn

class ResNet(nn.Module):
    """
    A residual neural network for the Connect6 game.

    Attributes:
    -----------
    game : Connect6Game
        An instance of the Connect6Game class.
    num_resBlocks : int
        The number of residual blocks in the network.
    num_hidden : int
        The number of hidden units in the convolutional layers.
    device : str
        The device to run the model on (default is 'cpu').

    Methods:
    --------
    forward(x)
        Performs a forward pass through the network.

    """
    def __init__(self, game, num_resBlocks, num_hidden, device = 'cpu'):
        """
        Initializes the ResNet class.

        Parameters:
        -----------
        game : Connect6Game
            An instance of the Connect6Game class.
        num_resBlocks : int
            The number of residual blocks in the network.
        num_hidden : int
            The number of hidden units in the convolutional layers.
        device : str
            The device to run the model on (default is 'cpu').

        """
        super().__init__()

        self.device = device

        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )

        self.to(device)
        
    def forward(self, x):
        """
        Performs a forward pass through the network.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor.

        Returns:
        --------
        policy : torch.Tensor
            The policy output tensor.
        value : torch.Tensor
            The value output tensor.

        """
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
        
    


class ResBlock(nn.Module):
    """
    A residual block module for a convolutional neural network.

    Args:
        num_hidden (int): The number of hidden units in the convolutional layers.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        bn1 (nn.BatchNorm2d): The first batch normalization layer.
        conv2 (nn.Conv2d): The second convolutional layer.
        bn2 (nn.BatchNorm2d): The second batch normalization layer.
    """
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


import math
import numpy as np

class AlphaMCTSNode:
    """
    A node in the AlphaMCTS tree.

    Attributes:
    - game: the game object
    - args: a dictionary of arguments
    - state: the state of the game
    - parent: the parent node
    - action_taken: the action taken to reach this node
    - prior: the prior probability of selecting this node
    - children: a list of child nodes
    - visit_count: the number of times this node has been visited
    - value_sum: the sum of values obtained from this node
    """

    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        """
        Initializes a new instance of the AlphaMCTSNode class.

        Parameters:
        - game: the game object
        - args: a dictionary of arguments
        - state: the state of the game
        - parent: the parent node
        - action_taken: the action taken to reach this node
        - prior: the prior probability of selecting this node
        - visit_count: the number of times this node has been visited
        """
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = []
        
        self.visit_count = visit_count
        self.value_sum = 0
        
    def is_fully_expanded(self):
        """
        Returns True if all possible child nodes have been created, False otherwise.
        """
        return len(self.children) > 0
    
    def select(self):
        """
        Selects the child node with the highest UCB score.

        Returns:
        - The selected child node.
        """
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        """
        Calculates the UCB score for a child node.

        Parameters:
        - child: the child node

        Returns:
        - The UCB score.
        """
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        """
        Expands the node by creating child nodes for each possible action.

        Parameters:
        - policy: a list of probabilities for each action

        Returns:
        - The newly created child node.
        """
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = AlphaMCTSNode(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)
                
        return child
            
    def backpropagate(self, value):
        """
        Backpropagates the value obtained from a simulation to the root node.

        Parameters:
        - value: the value obtained from a simulation
        """
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)




class AlphaMCTS:
    def __init__(self, game, args, model):
        """
        Initializes the AlphaMCTS class.

        Args:
        - game: an instance of the game class.
        - args: a dictionary containing the arguments for the AlphaMCTS algorithm.
        - model: the neural network model used for the AlphaMCTS algorithm.
        """
        self.game = game
        self.args = args
        self.model = model
    
    @torch.no_grad() #No calcular los gradientes (solo inferencia)
    def search(self, state):
        """
        Searches for the best move using the AlphaMCTS algorithm.

        Args:
        - state: the current state of the game.

        Returns:
        - action_probs: a numpy array containing the probability distribution over the possible actions.
        """
        root = AlphaMCTSNode(self.game, self.args, state, visit_count=1)
        
        policy, value = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        

        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)

        root.expand(policy)

        for search in range(self.args['num_searches']):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )

                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy = policy * valid_moves
                policy /= np.sum(policy)

                value = value.item()


                node = node.expand(policy)
                
                
            node.backpropagate(value)    
            
            
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
    



#--------------------------------------------------------------
class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = AlphaMCTS(game, args, model)

    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()
        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)
            memory.append([neutral_state, action_probs, player])

            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            action = np.random.choice(self.game.action_size, p=action_probs)

            state = self.game.get_next_state(state, action, player)
            value, is_terminal = self.game.get_value_and_terminated(state, action)
            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory
            player = self.game.get_opponent(player)   

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx: min(len(memory) - 1, batchIdx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            #Computar el loss total (policy + value)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            self.loss = policy_loss + value_loss


            # Optimizacion del modelo
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()





    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfPlay_iteration in tqdm(range(self.args['num_selfPlay_iterations'])):
                memory += self.selfPlay()
            self.model.train()

            for epoch in tqdm(range(self.args['num_epochs'])):
                self.train(memory)

            torch.save(self.model.state_dict(), f'./models/model_{iteration}_{self.game}.pt')
            torch.save(self.optimizer.state_dict(), f'./models/optimizer_{iteration}_{self.game}.pt')


#    ENTRENAMIENTO DEL MODELO
#--------------------------------------------------------------
# from tictactoe_game import TicTacToe

# ticTacToe = TicTacToe()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# player = 1
# args = {
#     'C': 1.41,
#     'num_searches': 60,
#     'num_iterations': 3,
#     'num_selfPlay_iterations': 500,
#     'num_epochs': 4,
#     'batch_size': 64,
#     'temperature': 1.25,
#     'dirichlet_epsilon': 0.25,
#     'dirichlet_alpha': 0.3
# }

# model = ResNet(ticTacToe, 4, 64, device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# alphaZero = AlphaZero(model, optimizer, ticTacToe, args)
# alphaZero.learn()

