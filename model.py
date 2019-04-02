import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from loss import MSEloss_with_Mask


def activation(input, type):
  
  if type.lower()=='selu':
    return F.selu(input)
  elif type.lower()=='elu':
    return F.elu(input)
  elif type.lower()=='relu':
    return F.relu(input)
  elif type.lower()=='relu6':
    return F.relu6(input)
  elif type.lower()=='lrelu':
    return F.lrelu(input)
  elif type.lower()=='tanh':
    return F.tanh(input)
  elif type.lower()=='sigmoid':
    return F.sigmoid(input)
  elif type.lower()=='swish':
    return F.sigmoid(input)*input
  elif type.lower()=='identity':
    return input
  else:
    raise ValueError("Unknown non-Linearity Type")


class AutoEncoder(nn.Module):

    def __init__(self, layer_sizes, nl_type='selu', is_constrained=True, dp_drop_prob=0.0, last_layer_activations=True):
        """
        layer_sizes = size of each layer in the autoencoder model
        For example: [10000, 1024, 512] will result in:
            - encoder 2 layers: 10000x1024 and 1024x512. Representation layer (z) will be 512
            - decoder 2 layers: 512x1024 and 1024x10000.
        
        nl_type = non-Linearity type (default: 'selu).
        is_constrained = If true then the weights of encoder and decoder are tied.
        dp_drop_prob = Dropout probability
        last_layer_activations = Whether to apply activation on last decoder layer
        """

        super(AutoEncoder, self).__init__()
        
