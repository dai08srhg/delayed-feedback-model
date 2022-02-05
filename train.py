import pandas as pd
from typing import List, Tuple
import torch
from torch import optim
from model.dfm_dataset import DfmDataset
from model.dfm import DelayedFeedbackModel
from model.dfm_loss import DfmLoss
from category_encoders import OrdinalEncoder


def load_dataset() -> pd.DataFrame:
    """
    Load dataset from local storage.

    Returns:
        pd.DataFrame: dataset
    """
    df = pd.read_pickle('/workspace/data/sample.pkl')
    return df


def preprocess(df: pd.DataFrame) -> Tuple[OrdinalEncoder, pd.DataFrame]:
    """
    Encoed categorical feature and negative example cv_delay_day is not used, so fill it with 0.

    Args:
        df: law dataset
    Returns:
        OrdinalEncoder: encoder
        pd.DataFrame: preprocessed dataset
    """
    categorical_columns = ['feature1', 'feature2', 'feature3']
    encoder = OrdinalEncoder(cols=categorical_columns, handle_unknown='impute').fit(df)
    # Encode
    df = encoder.transform(df)

    # Fill NaN in cv_delay_day with 0.
    df['cv_delay_day'] = df['cv_delay_day'].fillna(0)

    return encoder, df


def fix_params(model: DelayedFeedbackModel, layer: str) -> None:
    """
    Set requires_grad=False of logistic regression or hazard function

    Args:
        model (DelayedFeedbackModel): Pytorch model
        layer (str): 'logistic' or 'hazard'
    """
    if layer == 'logistic':
        for param in model.logistic.parameters():
            param.requires_grad = False
        for param in model.hazard_function.parameters():
            param.requires_grad = True

    if layer == 'hazard':
        for param in model.logistic.parameters():
            param.requires_grad = True
        for param in model.hazard_function.parameters():
            param.requires_grad = False


def get_embedding_size(df: pd.DataFrame, embedding_dim: int) -> List[Tuple[int, int]]:
    """
    Get embedding size

    Args:
        df (pd.DataFrame): Train dataset
        embedding_dim (int): Number of embedded dimensions
    Returns:
        List[Tuple[int, int]]: List of (Unique number of categories, embedding_dim)
    """
    # Extract feature columns
    df_feature = df.drop(columns=['supervised', 'elapsed_day', 'cv_delay_day'])

    # Get embedding layer size
    max_idxs = list(df_feature.max())
    embedding_sizes = []
    for i in max_idxs:
        embedding_sizes.append((int(i + 1), embedding_dim))

    return embedding_sizes


def train(df: pd.DataFrame):
    """
    Train delayed-feedback-model.
    Alternately optimize logistic regression and hazard function.

    Args:
        df (pd.DataFrame): preprocesed dataset
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Build dataset
    dataset = DfmDataset(df)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # Build model
    embedding_sizes = get_embedding_size(df, 5)
    model = DelayedFeedbackModel(embedding_sizes)

    epochs = 20
    loss_fn = DfmLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001)

    # Alternately optimize logistic regression and hazard function
    model.train()
    for epoch in range(epochs):
        # Optimize params of logistic regression
        fix_params(model, 'hazard')
        for X, y, elapsed_day, cv_delay_day in data_loader:

            X = X.to(device)
            y = y.to(device)
            elapsed_day = elapsed_day.to(device)
            cv_delay_day = cv_delay_day.to(device)

            # Initialize gradient
            optimizer.zero_grad()

            p, lam = model(X)
            loss = loss_fn(p, lam, y, elapsed_day, cv_delay_day)
            loss.backward()

            # Update
            optimizer.step()

        # Optimize params of hazard function
        fix_params(model, 'logistic')
        for X, y, elapsed_day, cv_delay_day in data_loader:

            X = X.to(device)
            y = y.to(device)
            elapsed_day = elapsed_day.to(device)
            cv_delay_day = cv_delay_day.to(device)

            # Initialize gradient
            optimizer.zero_grad()

            p, lam = model(X)
            loss = loss_fn(p, lam, y, elapsed_day, cv_delay_day)
            loss.backward()

            # Update
            optimizer.step()


def main():
    df = load_dataset()

    encoder, df = preprocess(df)

    train(df)


if __name__ == '__main__':
    main()
