import torch
import torch.nn as nn
from harl.utils.models_tools import init, get_active_func, get_init_method
from harl.models.base.mlp import MLPLayer
from harl.models.base.rnn import LSTMLayer

class MixingLayer(nn.Module):
    def __init__(self, method="average"):
        super(MixingLayer, self).__init__()
        self.method = method.lower()
        if self.method not in ["average"]:
            raise ValueError(f"Unsupported mixing method: {self.method}")
    
    def forward(self, *inputs):
        if len(inputs) == 0:
            raise ValueError("MixingLayer requires at least one input.")

        # Ensure all inputs have the same dimensions
        first_shape = inputs[0].shape
        for i, tensor in enumerate(inputs):
            if tensor.shape != first_shape:
                raise ValueError(f"Input {i} shape {tensor.shape} does not match first input shape {first_shape}.")

        # Apply the mixing method
        if self.method == "average":
            return self._average(inputs)
        else:
            raise NotImplementedError(f"Mixing method '{self.method}' is not implemented.")
    
    def _average(self, inputs):
        """Compute the element-wise mean of the inputs."""
        stacked = torch.stack(inputs, dim=0)  # Stack along a new dimension
        return torch.mean(stacked, dim=0)    # Compute mean along the new dimension


class ActivateGate(nn.Module):
    def __init__(self, input_dim, hidden_sizes, initialization_method, activation_func, threshold=0.5):

        super(ActivateGate, self).__init__()
        self.mlp = MLPLayer(
            input_dim,
            hidden_sizes,
            initialization_method,
            activation_func
        )
        self.final_layer = nn.Linear(hidden_sizes[-1], 1)  # Single scalar output
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation to constrain output to [0, 1]
        self.threshold = threshold  # Threshold for binary output

    def forward(self, x):

        x = self.mlp(x)  # Pass through the MLP
        x = self.final_layer(x)  # Reduce to a single scalar
        x = self.sigmoid(x)  # Constrain to [0, 1]
        binary_output = (x > self.threshold).float()  # Threshold to binary output
        return binary_output

class Neuron(nn.Module):
    """
    A neuron module that processes input data and updates its internal state.
    """

    def __init__(self, args, obs_shape):
        super(Neuron, self).__init__()

        self.use_feature_normalization = args["use_feature_normalization"]
        self.initialization_method = args["initialization_method"]
        self.activation_func = args["activation_func"]
        self.hidden_sizes = args["hidden_sizes"]
        self.recurrent_n = args["recurrent_n"]

        obs_dim = obs_shape[0]

        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mixing_layer = MixingLayer(method="average")

        self.lstm = LSTMLayer(
            self.hidden_sizes[-1],
            self.hidden_sizes[-1],
            self.recurrent_n,
            self.initialization_method,
        )

        self.activation_gate = ActivateGate(
            self.hidden_sizes[-1],
            self.hidden_sizes,
            self.initialization_method,
            self.activation_func,
            threshold=0.5
        )

        self.processing_layer = MLPLayer(
            self.hidden_sizes[-1],
            self.hidden_sizes,
            self.initialization_method,
            self.activation_func
        )

    def forward(self, *x, hxs, masks):
        x = self.mixing_layer(*x)

        if self.use_feature_normalization:
            x = self.feature_norm(x)
        
        x, hxs = self.lstm(x, hxs, masks)

        activation = self.activation_gate(x)

        # hxs = hxs * (1-activation)

        x = self.processing_layer(x)*activation

        return x, hxs
