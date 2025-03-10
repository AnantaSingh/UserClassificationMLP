import torch
import torch.nn as nn

class UserClassifierMLP(nn.Module):
    def __init__(self, input_size=6):
        super(UserClassifierMLP, self).__init__()
        
        # Single hidden layer with 2 nodes
        self.model = nn.Sequential(
            nn.Linear(input_size, 2),
            nn.ReLU(),
            nn.Linear(2, 2)
        )
        
        # Basic initialization
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.model(x)
    
    def save_weights(self, path):
        """Save model weights to file."""
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path):
        """Load model weights from file."""
        self.load_state_dict(torch.load(path)) 