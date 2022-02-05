import torch
import torch.nn as nn


class DfmLoss(nn.Module):
    """
    Loss function of delayed-feedback-model
    """

    def __init__(self):
        super().__init__()

    def forward(self, p, lam, y, elapsed_day, cv_delay_day):
        '''
        Args:
            p: output of logistic regression
            lam: output of hazard function
            y: conversion label
            elapsed_day: days elapsed since click
            cv_delay_day: days delayed from click to conversion
        '''
        # size=(minibatch) -> size=(1, minibatch)
        y = torch.unsqueeze(y, 1)
        p = torch.unsqueeze(p, 1)
        elapsed_day = torch.unsqueeze(elapsed_day, 1)
        cv_delay_day = torch.unsqueeze(cv_delay_day, 1)

        # loss of positive instance
        p_loss = -y * (torch.log(p) + torch.log(lam) - (lam * cv_delay_day))
        # loss of negative instance
        n_loss = -(1 - y) * torch.log(1 - p + p * torch.exp(-lam * elapsed_day))

        loss = torch.mean(p_loss + n_loss)

        return loss
