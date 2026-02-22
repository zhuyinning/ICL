import torch
from types import SimpleNamespace
from torch_geometric.loader import DataLoader

def _print_batch(batch):
    print("batch.x:", None if batch.x is None else tuple(batch.x.shape))
    print("batch.edge_index:", tuple(batch.edge_index.shape))
    print("batch.y:", None if batch.y is None else tuple(batch.y.shape))
    print("batch.batch:", tuple(batch.batch.shape))

def check_molhiv(root="./data_dir"):
    from ogb.graphproppred import PygGraphPropPredDataset

    name = "ogbg-molhiv"
    print(f"\n===== Checking {name} =====")

    dataset = PygGraphPropPredDataset(name=name, root=root)
    print("num_graphs =", len(dataset))
    print("num_tasks  =", dataset.num_tasks)
    print("example    =", dataset[0])

    split_idx = dataset.get_idx_split()
    print("split sizes:",
          "train =", len(split_idx["train"]),
          "valid =", len(split_idx["valid"]),
          "test  =", len(split_idx["test"]))

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    _print_batch(batch)

    print(f"[OK] {name}")

def check_mnistsp():
    print("\n===== Checking MNIST-75sp (mnistsp) =====")

    from datasets import get_mnistsp
    args = SimpleNamespace()
    args.batch_size = 32

    train_dataset, val_dataset, test_dataset = get_mnistsp(args)
    print("split sizes:",
          "train =", len(train_dataset),
          "val =", len(val_dataset),
          "test =", len(test_dataset))

    g0 = train_dataset[0]
    print("example =", g0)

    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    _print_batch(batch)

    print("[OK] mnistsp")

def check_sst():
    print("\n===== Checking Graph-SST2 (sst) =====")

    from datasets import get_sst
    args = SimpleNamespace()
    args.batch_size = 32

    train_dataset, val_dataset, test_dataset = get_sst(args)
    print("split sizes:",
          "train =", len(train_dataset),
          "val =", len(val_dataset),
          "test =", len(test_dataset))

    g0 = train_dataset[0]
    print("example =", g0)

    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    _print_batch(batch)

    print("[OK] sst")

def main():
    # Run in this order (from easiest to hardest)
    check_molhiv()
    check_mnistsp()
    check_sst()

if __name__ == "__main__":
    main()
