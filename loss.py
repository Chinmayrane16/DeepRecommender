import torch
import torch.nn as nn
from torch.autograd import Variable


def MSEloss_with_Mask(inputs, target):
    # Masking into a vector of 1's and 0's.
  mask = [int(i) for i in target]
  mask = torch.tensor(mask)

  # Actual number of ratings.
  number_ratings = torch.sum(mask)
  # To avoid division by zero while calculating loss.
  number_ratings = 1 if number_ratings == 0 else number_ratings.float()
  
  error = torch.sum(torch.mul(mask,torch.mul((target-inputs),(target-inputs))))
  loss = error.div(number_ratings)
  return loss
