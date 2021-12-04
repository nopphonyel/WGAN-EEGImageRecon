import os
import pickle
from torch.utils.data import Dataset, DataLoader

__dir_name__ = os.path.dirname(__file__)


class EMNIST_Extended(Dataset):
    __TRAIN_LOC = 'content/EMNIST/train.pkl'
    __TEST_LOC = 'content/EMNIST/test.pkl'

    def __init__(self, train: bool = True):
        super(EMNIST_Extended, self).__init__()
        p = os.path.join(__dir_name__, self.__TRAIN_LOC)
        self.ds = pickle.load(open(p, 'rb'))
        self.d = self.ds['data']
        self.l = self.ds['label']
        self.dev = 'cpu'

    def __len__(self):
        return self.d.shape[0]

    def __getitem__(self, idx):
        return self.d[idx, :, :, :].to(self.dev), self.l[idx].to(self.dev)

    def to(self, dev):
        self.dev = dev
        return self


if __name__ == "__main__":
    ds = EMNIST_Extended(train=True)
    ld = DataLoader(ds, batch_size=4, shuffle=True)
    for d, l in ld:
        print(d.shape, l.shape)
        print(l.dtype)
