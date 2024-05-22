# From https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks

import numpy as np
from torch.utils.data import DataLoader


class O5Synthetic(object):
    def __init__(self, N=1024):
        super().__init__()
        d = 5
        self.dim = 2 * d
        self.X = np.random.randn(N, self.dim) # B x 10
        ri = self.X.reshape(-1, 2, 5)    # B x 2 x d
        r1, r2 = ri.transpose(1, 0, 2)   # B x d and B x d 
        self.Y = (
            np.sin(np.sqrt((r1**2).sum(-1)))
            - 0.5 * np.sqrt((r2**2).sum(-1)) ** 3
            + (r1 * r2).sum(-1)
            / (np.sqrt((r1**2).sum(-1)) * np.sqrt((r2**2).sum(-1)))
        ) # B
        self.Y = self.Y[..., None] # B x 1
        # One has to be careful computing mean and std in a way so that standardizing
        # does not violate equivariance
        Xmean = self.X.mean(0)  # can add and subtract arbitrary tensors     # .shape = [10]
        Xscale = (
            np.sqrt((self.X.reshape(N, 2, d) ** 2).mean((0, 2)))[:, None] + 0 * ri[0]
        ).reshape(self.dim)
        self.stats = 0, Xscale, self.Y.mean(axis=0), self.Y.std(axis=0)
                                # .shape = [1]            .shape = [1]

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]

NUM_TEST = 16384

class O5Dataset:
    def __init__(self, num_samples=1024, batch_size=32):
        super().__init__()
        self.train_dataset = O5Synthetic(num_samples)
        self.val_dataset = O5Synthetic(NUM_TEST)
        self.test_dataset = O5Synthetic(NUM_TEST)

        self.batch_size = batch_size

        self.ymean, self.ystd = self.train_dataset.stats[-2].item(), self.train_dataset.stats[-1].item()

        self._normalize_datasets()

        # self._save_datasets()

        # exit()


    def _normalize_datasets(self):
        Xmean, Xscale, Ymean, Ystd = self.train_dataset.stats
        self.train_dataset.X -= Xmean
        self.train_dataset.X /= Xscale
        self.train_dataset.Y -= Ymean
        self.train_dataset.Y /= Ystd

        self.val_dataset.X -= Xmean
        self.val_dataset.X /= Xscale
        self.val_dataset.Y -= Ymean
        self.val_dataset.Y /= Ystd

        self.test_dataset.X -= Xmean
        self.test_dataset.X /= Xscale
        self.test_dataset.Y -= Ymean
        self.test_dataset.Y /= Ystd


    def _save_datasets(self):
        train_X = self.train_dataset.X.reshape(-1, 2, 5) # B x 2 x d 
        train_Y = self.train_dataset.Y
        print('\n\nmax train ||x1||, ||x2||:',  self._compute_max_norm(train_X))

        val_X = self.val_dataset.X.reshape(-1, 2, 5)     # B x 2 x d 
        val_Y = self.val_dataset.Y
        print('\n\nmax val ||x1||, ||x2||:',  self._compute_max_norm(val_X))

        test_X = self.test_dataset.X.reshape(-1, 2, 5)   # B x 2 x d
        test_Y = self.test_dataset.Y
        print('\n\nmax test ||x1||, ||x2||:',  self._compute_max_norm(test_X))

        # save as numpy arrays
        np.save('train_X.npy', train_X)
        np.save('train_Y.npy', train_Y)

        np.save('val_X.npy', val_X)
        np.save('val_Y.npy', val_Y)

        np.save('test_X.npy', test_X)
        np.save('test_Y.npy', test_Y)

        del train_X, train_Y, val_X, val_Y, test_X, test_Y


    def _compute_max_norm(self, X):
        import torch
        norm_X = torch.tensor(X).norm(dim=-1)
        max_x1 = norm_X[:,0].amax()
        max_x2 = norm_X[:,1].amax()
        return max_x1.item(), max_x2.item()


    def train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_loader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

    def test_loader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )