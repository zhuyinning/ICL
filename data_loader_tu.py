import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

def load_tu_dataset(name: str, root: str = "./data_tu"):
    dataset = TUDataset(root=root, name=name)

    # 强制补齐 x
    for i in range(len(dataset)):
        data = dataset.get(i) 
        if data.x is None:
            data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)

    return dataset

def make_loader(dataset, batch_size: int = 32, shuffle: bool = True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
