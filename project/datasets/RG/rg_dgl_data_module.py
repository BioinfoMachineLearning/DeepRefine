from typing import Optional

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from project.datasets.RG.rg_dgl_dataset import RGDGLDataset
from project.utils.deeprefine_utils import dgl_collate, rotate_randomly


class RGDGLDataModule(LightningDataModule):
    """Random graph data module for DGL with PyTorch."""

    # Dataset partition instantiations
    rg_train = None
    rg_val = None
    rg_test = None

    def __init__(self, n_lb=10, n_hb=20, e_lb=10, e_hb=15, num_node_features=6, num_pos_features=3,
                 num_coord_features=3, num_edge_features=4, num_fourier_features=0, size=300, out_dim=1,
                 train_transform=None, test_transform=rotate_randomly, dtype=np.float32, batch_size=4,
                 num_dataloader_workers=4):
        super().__init__()

        # Dataset parameters
        self.n_lb = n_lb
        self.n_hb = n_hb
        self.e_lb = e_lb
        self.e_hb = e_hb
        self.num_node_features = num_node_features
        self.num_pos_features = num_pos_features
        self.num_coord_features = num_coord_features
        self.num_edge_features = num_edge_features
        self.num_fourier_features = num_fourier_features
        self.size = size
        self.out_dim = out_dim
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.dtype = dtype

        # Dataset meta-parameters
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers

    def prepare_data(self):
        # Download the full dataset - called only on 1 GPU
        self.rg_train = RGDGLDataset(n_lb=self.n_lb, n_hb=self.n_hb, e_lb=self.e_lb, e_hb=self.e_hb,
                                     node_feature_size=self.num_node_features, edge_feature_size=self.num_edge_features,
                                     size=self.size, out_dim=self.out_dim, transform=self.train_transform,
                                     dtype=self.dtype)

    def setup(self, stage: Optional[str] = None):
        # Assign training/validation/testing data set for use in DataLoaders - called on every GPU
        self.rg_train = RGDGLDataset(n_lb=self.n_lb, n_hb=self.n_hb, e_lb=self.e_lb, e_hb=self.e_hb,
                                     node_feature_size=self.num_node_features, edge_feature_size=self.num_edge_features,
                                     size=self.size, out_dim=self.out_dim, transform=self.train_transform,
                                     dtype=self.dtype)
        self.rg_val = RGDGLDataset(n_lb=self.n_lb, n_hb=self.n_hb, e_lb=self.e_lb, e_hb=self.e_hb,
                                   node_feature_size=self.num_node_features, edge_feature_size=self.num_edge_features,
                                   size=self.size, out_dim=self.out_dim, transform=self.test_transform,
                                   dtype=self.dtype)
        self.rg_test = RGDGLDataset(n_lb=self.n_lb, n_hb=self.n_hb, e_lb=self.e_lb, e_hb=self.e_hb,
                                    node_feature_size=self.num_node_features, edge_feature_size=self.num_edge_features,
                                    size=self.size, out_dim=self.out_dim, transform=self.test_transform,
                                    dtype=self.dtype)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.rg_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_dataloader_workers, collate_fn=dgl_collate)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.rg_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=dgl_collate)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.rg_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=dgl_collate)
