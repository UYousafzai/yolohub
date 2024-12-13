# want to write some base_model level definitions

import torch
import abc

class BaseModel(abc.ABC, torch.nn.Module):
    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def fuse(self):
        pass
