from torch.utils.data import Dataset

class XYDataset(Dataset):
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        self.size = len(dataFrame)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            'sequences': self.dataFrame['sequence'].iloc[idx],
            'label_class': self.dataFrame['label_class'].iloc[idx],      
            'label_activity': self.dataFrame['label_activity'].iloc[idx]
        }


class XDataset(Dataset):
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        self.size = len(dataFrame)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {'sequences': self.dataFrame['sequence'].iloc[idx]}

