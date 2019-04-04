import torch
import torch.nn as nn
from torch.autograd import Variable


def MSEloss_with_Mask(inputs, target):
    # Masking into a vector of 1's and 0's.
  mask = [float(i!=0.0) for i in target]
  mask = torch.tensor(mask)

  # Actual number of ratings.
  number_ratings = torch.sum(mask)
  # To avoid division by zero while calculating loss.
  number_ratings = 1.0 if number_ratings == 0.0 else number_ratings
  
  error = torch.sum(torch.mul(mask,torch.mul((target-inputs),(target-inputs))))
  loss = error.div(number_ratings)
  return loss
