from torch import nn
import torch
import torch.nn.functional as F


class MyNeuralNet(torch.nn.Module):
    """ Basic neural network class. 
    
    Args:
        in_features: number of input features
        out_features: number of output features
    
    """
    def __init__(self) -> None:
        # self.l1 = torch.nn.Linear(in_features, 500)
        # self.l2 = torch.nn.Linear(500, out_features)
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)  
        self.r = torch.nn.ReLU()
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        x = x.view(x.shape[0], -1)
        x = self.dropout(self.r(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x))) #same as the other lines just a different syntax
        x = self.dropout(self.r(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        # return self.l2(self.r(self.l1(x)))
        return x