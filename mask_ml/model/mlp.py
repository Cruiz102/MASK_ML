import torch
import torch.nn as nn
import torch.nn.functional as F 
from typing import Optional, List



class MLPClassification(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        input_size: int,
        num_layers: int,
        sigmoid_output: bool
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.input_size = input_size
        self.num_layers = num_layers
        self.sigmoid_output = sigmoid_output
        h = [hidden_size] * (num_layers - 1)
        self.flat = nn.Flatten()
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_size] + h, h + [num_classes])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        x = self.flat(x)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x



class MLP(nn.Module):
    def __init__(self, activation_function: str = "relu", 
                input_size: int = 32 ,
                output_size: int = 10,
                hidden_sizes: Optional[List[int]] = None):
        super(MLP, self).__init__()
        if hidden_sizes:
            layer_sizes = [input_size] + hidden_sizes + [output_size]
        else:
            layer_sizes = [input_size, output_size]
        
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        if activation_function == "gelu":
            self.activation = nn.GELU()
        elif activation_function == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation_function == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass input through all layers except the last one with activation
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        
        x = self.layers[-1](x)
        return x
