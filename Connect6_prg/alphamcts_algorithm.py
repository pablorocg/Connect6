import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(0)


class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, numHidden):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, numHidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(numHidden),
            nn.ReLU(),
        )

        self.backbone = nn.ModuleList(
            [ResBlock(numHidden) for _ in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(numHidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(numHidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.startBlock(x)
        for block in self.backbone:
            x = block(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
    

class ResBlock(nn.Module):
    def __init__(self, numHidden):
        super().__init__()
        
        self.conv1 = nn.Conv2d(numHidden, numHidden, kernel_size=3, padding=1),
        self.bn1 = nn.BatchNorm2d(numHidden),
        self.conv2 = nn.Conv2d(numHidden, numHidden, kernel_size=3, padding=1),
        self.bn2 = nn.BatchNorm2d(numHidden)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
    

class AlphaMCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        
    def search(self, state):
        root = Node(self.game, self.args, state)
        
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