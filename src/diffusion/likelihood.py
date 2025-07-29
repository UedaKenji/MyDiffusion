# from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

__OPERATOR__ = {}


def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls

    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass

    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name="noise")
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device

    def forward(self, data):
        return data

    def transpose(self, data):
        return data

    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


class Likelihood(ABC):
    def __init__(self, nn_model, device, y_sigma=None):
        self.nn_model = nn_model
        # self.measurement = measurement
        self.device = device
        # self.y_sigma = y_sigma

    def x_grad(self, x, sigma):
        x = x.to(self.device)
        x.requires_grad = True
        x_hat = self.x_hat(x, sigma)
        log_likelihood = self.log_likelihood(x_hat, sigma)
        x_grad = torch.autograd.grad(
            outputs=log_likelihood,
            inputs=x,
            grad_outputs=torch.ones_like(log_likelihood),
            create_graph=False,
        )[0]
        x = x.detach_()
        return x_grad

    def x_hat(self, x, sigma):
        x_hat = x - sigma * self.nn_model(x, sigma * torch.ones(x.shape[0], 1).to(self.device))

        return x_hat

    @abstractmethod
    def y_estimated_sigma_sq(self, sgima_t, **kwargs):
        pass

    @abstractmethod
    def log_likelihood(self, x_hat, sigma=None, **kwargs):
        pass
