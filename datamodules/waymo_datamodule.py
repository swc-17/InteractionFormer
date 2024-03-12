
from typing import Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from datasets import WaymoDataset
from utils.target_builder import TargetBuilder, WaymoTargetBuilder


class WaymoDataModule(pl.LightningDataModule):
    transforms = {
        "TargetBuilder": TargetBuilder,
        "WaymoTargetBuilder": WaymoTargetBuilder,
    }

    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 test_batch_size: int,
                 shuffle: bool = False,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_raw_dir: Optional[str] = None,
                 val_raw_dir: Optional[str] = None,
                 test_raw_dir: Optional[str] = None,
                 train_processed_dir: Optional[str] = None,
                 val_processed_dir: Optional[str] = None,
                 test_processed_dir: Optional[str] = None,
                 transform: Optional[str] = None,
                 cluster: bool = False,
                 **kwargs) -> None:
        super(WaymoDataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.train_processed_dir = train_processed_dir
        self.val_processed_dir = val_processed_dir
        self.test_processed_dir = test_processed_dir
        if transform is None:
            train_transform = TargetBuilder(11, 80)
        else:
            train_transform = WaymoDataModule.transforms[transform](11, 80, "train")
        if transform is None:
            val_transform = TargetBuilder(11, 80)
        else:
            val_transform = WaymoDataModule.transforms[transform](11, 80, "val")
        if transform is None:
            test_transform = TargetBuilder(11, 80)
        else:
            test_transform = WaymoDataModule.transforms[transform](11, 80)

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.cluster = cluster

    def prepare_data(self) -> None:
        WaymoDataset(self.root, 'train', processed_dir=self.train_processed_dir,
                     transform=self.train_transform, cluster=self.cluster)
        WaymoDataset(self.root, 'val', processed_dir=self.val_processed_dir,
                     transform=self.val_transform, cluster=self.cluster)
        WaymoDataset(self.root, 'test', processed_dir=self.test_processed_dir,
                     transform=self.test_transform, cluster=self.cluster)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = WaymoDataset(self.root, 'train', processed_dir=self.train_processed_dir,
                                          transform=self.train_transform, cluster=self.cluster)
        self.val_dataset = WaymoDataset(self.root, 'val', processed_dir=self.val_processed_dir,
                                        transform=self.val_transform, cluster=self.cluster)
        self.test_dataset = WaymoDataset(self.root, 'test', processed_dir=self.test_processed_dir,
                                         transform=self.test_transform, cluster=self.cluster)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)
