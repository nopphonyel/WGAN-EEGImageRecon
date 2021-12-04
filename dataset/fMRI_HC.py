import pickle
import os
from torch.utils.data import Dataset, DataLoader


class fMRI_HC_Dataset(Dataset):
    __DAT_PATH = 'content/DSC_2018.00114_120_v1/rebuild'
    __IMG_SIZE = 64
    __DIR_NAME = os.path.dirname(__file__)
    __dev = 'cpu'

    def __init__(self, p_id: int, v: int = 1, train: bool = True):
        """
        :param p_id: Participant ID
        :param v: Version (only V1 and V2 available)
        :param train: Using train data or not
        """
        f_name = fMRI_HC_Dataset.__f_name_gen(p_id, v, train)
        ld_path = os.path.join(
            fMRI_HC_Dataset.__DIR_NAME,
            fMRI_HC_Dataset.__DAT_PATH,
            f_name
        )
        loader = pickle.load(open(ld_path, "rb"))
        self.fmri = loader['X']
        self.img = loader['Y']
        self.label = loader['L']

    def to(self, dev):
        self.__dev = dev
        return self

    def __len__(self):
        return self.fmri.shape[0]

    def __getitem__(self, idx):
        fmri = self.fmri[idx, :].to(self.__dev)
        img = self.img[idx, :].to(self.__dev)
        label = self.label[idx].to(self.__dev)
        return fmri, img, label

    @staticmethod
    def __f_name_gen(p_id: int, v: int, train: bool):
        f_name = "XS%02d_V%01d" % (p_id, v)
        if train:
            f_name += "_{}x{}_train.dat".format(fMRI_HC_Dataset.__IMG_SIZE, fMRI_HC_Dataset.__IMG_SIZE)
        else:
            f_name += "_{}x{}_test.dat".format(fMRI_HC_Dataset.__IMG_SIZE, fMRI_HC_Dataset.__IMG_SIZE)
        return f_name

    @staticmethod
    def get_name():
        return "fMRI_HC_Dataset"

    def get_num_classes(self):
        return 6

## Testing fragment
# import matplotlib.pyplot as plt
#
# if __name__ == '__main__':
#     ds = fMRI_HC_Dataset(p_id=3, v=1, train=True)
#     ld = DataLoader(ds, batch_size=2)
#     for idx, (fmri, img, label) in enumerate(ld):
#         if idx == 16:
#             print(label[0])
#             plt.imshow(img[0])
#             plt.show()
