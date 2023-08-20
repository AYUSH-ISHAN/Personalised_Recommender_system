import torch
import torch.nn as nn

from .classifiers import register
from ..modules import *


__all__ = ['LogisticClassifier']


@register('logistic')
class LogisticClassifier(Module):
  def __init__(self, in_dim=196, n_way=1, temp=1., learn_temp=False):
    super(LogisticClassifier, self).__init__()
    self.in_dim = in_dim
    self.n_way = n_way
    self.temp = temp
    self.learn_temp = learn_temp
    # print("class : ", in_dim.shape, n_way)
    self.linear = Linear(196, 1)
    if self.learn_temp:
      self.temp = nn.Parameter(torch.tensor(temp))

  def reset_parameters(self):
    nn.init.zeros_(self.linear.weight)
    nn.init.zeros_(self.linear.bias)

  def forward(self, x_shot, params=None):
    # assert x_shot.dim() == 2
    # print(get_child_dict(params, 'linear'))
    logits = self.linear(x_shot, get_child_dict(params, 'linear'))
    logits = logits * self.temp
    return logits