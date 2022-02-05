import torch
import pandas as pd


class DfmDataset(torch.utils.data.Dataset):
    """
    Dataset of Delayed Feedback Model (DFM)
    """

    def __init__(self, df: pd.DataFrame):
        df_feature = df.drop(columns=['supervised', 'elapsed_day', 'cv_delay_day'])

        self.X = torch.from_numpy(df_feature.values).long()
        self.y = torch.from_numpy(df['supervised'].values)
        self.elapsed_day = torch.from_numpy(df['elapsed_day'].values)
        self.cv_delay_day = torch.from_numpy(df['cv_delay_day'].values)

        self.data_num = len(self.X)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.elapsed_day[idx], self.cv_delay_day[idx]
