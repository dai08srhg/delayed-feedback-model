import torch
import torch.nn as nn
from typing import List, Tuple


class FeatureExtractor(nn.Module):
    """
    Embedding layer for encoding categorical variables.
    """

    def __init__(self, embedding_sizes: List[Tuple[int, int]]):
        """
        Args:
            embedding_sizes (List[Tuple[int, int]]): List of (Unique categorical variables + 1, embedding dim)
        """
        super(FeatureExtractor, self).__init__()
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(unique_size, embedding_dim) for unique_size, embedding_dim in embedding_sizes])

    def forward(self, category_inputs):
        # Embedding each variable
        h = [embedding_layer(category_inputs[:, i]) for i, embedding_layer in enumerate(self.embedding_layers)]
        # Concat each vector
        h = torch.cat(h, dim=1)  # size = (minibath, embeding_dim * Number of categorical variables)
        return h


class LogisticRegression(nn.Module):
    """
    Logistic Regression for conversion prediction
    """

    def __init__(self, input_dim: int):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        h = self.linear(inputs)
        p = self.sigmoid(h)
        return p


class HazardFunction(nn.Module):
    """
    hazard function λ
    """

    def __init__(self, input_dim: int):
        super(HazardFunction, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, inputs):
        lam = torch.exp(self.linear(inputs))
        return lam


class DelayedFeedbackModel(nn.Module):
    """
    delayed feedback model.
    Consists of embedding layer, logistic function, hazard function.
    """

    def __init__(self, embedding_sizes: List[Tuple[int, int]]):
        """
        Args:
            embedding_sizes (List[Tuple[int, int]]): List of (Unique categorical variables + 1, embedding dim)
        """
        super(DelayedFeedbackModel, self).__init__()
        self.feature_extractor = FeatureExtractor(embedding_sizes)
        input_dim = 0
        for _, embedding_dim in embedding_sizes:
            input_dim += embedding_dim

        self.logistic = LogisticRegression(input_dim)
        self.hazard_function = HazardFunction(input_dim)

    def train_logistic_mode(self):
        """
        Train logistic regression mode.
        Set requires_grad of logistic regression to True and hazard function to False.
        """
        for param in self.logistic.parameters():
            param.requires_grad = True
        for param in self.hazard_function.parameters():
            param.requires_grad = False

    def train_hazard_function_mode(self):
        """
        Train hazard function mode.
        Set requires_grad of hazard function to True and logistic regression to False.
        """
        for param in self.logistic.parameters():
            param.requires_grad = False
        for param in self.hazard_function.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        h = self.feature_extractor(inputs)  # embedding
        p = self.logistic(h)
        lam = self.hazard_function(h)

        return p, lam
