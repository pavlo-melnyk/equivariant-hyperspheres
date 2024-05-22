import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset

from pytorch3d_transforms import Transform3d, random_rotations
from pathlib import Path
from dataclasses import dataclass



def make_object(**kwargs):

    cls = kwargs['cls']
    kwargs = {k: v for k, v in kwargs.items() if k != 'cls'}
    obj = cls(**kwargs)
    return obj


@dataclass
class PointcloudRandomTransform1:
    rot: str = 'I'  # Rotation variant (z, so3 or I (or aligned), or o3 (with reflections))
    scale: bool = False  # Whether to apply random (0.66 - 1.5) scaling after rotation
    trans: bool = False  # Whether to apply random (-0.2 - 0.2) translation after scaling
    shuffle: bool = False  # Whether to shuffle the points after transforming them
    num_points: int = 1024

    # if center and normalize to a unit sphere:
    center_normalize: bool = False

    def __call__(self, pcd: torch.Tensor):

        # if self.center_normalize:

        T = Transform3d(device=pcd.device)

        if self.scale:
            T = T.scale(*np.random.uniform(low=2. / 3., high=3. / 2., size=[3]))

        if self.trans:
            T = T.translate(*np.random.uniform(low=-0.2, high=0.2, size=[3]))

        if self.rot == 'x':
            T = T.rotate_axis_angle(angle=torch.rand(1) * 360, axis="X", degrees=True)
        elif self.rot == 'y':
            T = T.rotate_axis_angle(angle=torch.rand(1) * 360, axis="Y", degrees=True)
        elif self.rot == 'z':
            T = T.rotate_axis_angle(angle=torch.rand(1) * 360, axis="Z", degrees=True)
        elif self.rot == 'so3':
            T = T.rotate(R=random_rotations(1))
        elif self.rot == 'o3':
            T = T.rotate(R=random_rotations(1))
            # select a reflection axis:
            xyz = np.array([-1., 1., 1.])
            np.random.shuffle(xyz)
            T = T.scale(*xyz)
        elif self.rot == 'I' or self.rot == 'aligned':
            pass
        else:
            raise ValueError

        pcd = T.transform_points(pcd)

        if len(pcd.shape) == 2:
            pcd = pcd[:self.num_points]
        else:
            # len(pcd.shape) == 3
            pcd = pcd[:, :self.num_points]

        if self.shuffle:
            pcd = pcd[torch.randperm(pcd.shape[0])]

        return pcd
    

class UTKinectSkeletons(Dataset):
    label_names = ['sitDown', 'throw', 'carry', 'push', 'waveHands',
                   'walk', 'clapHands', 'pull', 'pickUp', 'standUp']

    def __init__(self, partition='train', transform: callable = None, dset_root=None):
        super().__init__()

        self.partition = partition
        self.transform = transform

        # dset_root = Path.home() / dset_root
        dset_root = Path(dset_root)
        
        self.data = torch.from_numpy(np.load(dset_root / f"utkinect_skeletons/X{partition}.npy")).float()
        self.label = torch.from_numpy(np.load(dset_root / f"utkinect_skeletons/Y{partition}.npy"))

    @property
    def num_classes(self):
        return len(self.label_names)

    def __getitem__(self, item):
        pcd = self.data[item]
        label = self.label[item]

        if self.transform is not None:
            pcd = self.transform(pcd)

        return pcd, label.squeeze()

    def __len__(self):
        return self.data.shape[0]


class UTKinectSkeletonsDataset:

    def __init__(self, rot="I", batch_size=32, add_to_train_partition='', dset_root=None):
        super().__init__()

        tform_args = dict(train=dict(cls=PointcloudRandomTransform1, rot=rot, scale=False, trans=False, shuffle=False),
                          val=dict(cls=PointcloudRandomTransform1, rot='o3', scale=False, trans=False, shuffle=False),
                          test=dict(cls=PointcloudRandomTransform1, rot='o3', scale=False, trans=False, shuffle=False))

        self.train_dataset = UTKinectSkeletons(partition='train'+add_to_train_partition, transform=make_object(**tform_args["train"]), dset_root=dset_root)
        self.val_dataset = UTKinectSkeletons(partition='val', transform=make_object(**tform_args["val"]), dset_root=dset_root)
        self.test_dataset = UTKinectSkeletons(partition='test', transform=make_object(**tform_args["test"]), dset_root=dset_root)

        self.batch_size = batch_size

    def train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_loader(self):
        return DataLoader(
            self.val_dataset, batch_size=32, shuffle=False, drop_last=False
        )

    def test_loader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            drop_last=False,
        )
