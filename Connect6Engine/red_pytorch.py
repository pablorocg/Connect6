# Crear el codigo de la red neuronal en pytorch para MNIST

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    """
    A neural network model for Connect6 game.

    Attributes:
    -----------
    conv1 : torch.nn.Conv2d
        First convolutional layer.
    conv2 : torch.nn.Conv2d
        Second convolutional layer.
    fc1 : torch.nn.Linear
        First fully connected layer.
    fc2 : torch.nn.Linear
        Second fully connected layer.
    optimizer : torch.optim.SGD
        Stochastic gradient descent optimizer.
    criterion : torch.nn.CrossEntropyLoss
        Cross entropy loss function.

    Methods:
    --------
    forward(x: torch.Tensor) -> torch.Tensor:
        Forward pass of the neural network.
    train(*args, **kwargs) -> None:
        Train the neural network.
    test(*args, **kwargs) -> None:
        Test the neural network.
    predict(x: torch.Tensor) -> torch.Tensor:
        Predict the output of the neural network.
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the neural network.

        Parameters:
        -----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor.

        Returns:
        --------
        torch.Tensor
            Output tensor.
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def train(self, *args, **kwargs) -> None:
        """
        Train the neural network.

        Parameters:
        -----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        super().train(*args, **kwargs)
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def test(self, *args, **kwargs) -> None:
        """
        Test the neural network.

        Parameters:
        -----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        super().test(*args, **kwargs)
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the output of the neural network.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor.

        Returns:
        --------
        torch.Tensor
            Output tensor.
        """
        return self.forward(x)
    