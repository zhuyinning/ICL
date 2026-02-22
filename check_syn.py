import torch
from types import SimpleNamespace
from torch_geometric.loader import DataLoader
from datasets import get_SPMotif


def print_batch(batch):
    print("batch.x:", None if batch.x is None else tuple(batch.x.shape))
    print("batch.edge_index:", tuple(batch.edge_index.shape))
    print("batch.y:", None if batch.y is None else tuple(batch.y.shape))
    print("batch.batch:", tuple(batch.batch.shape))


def check_spmotif(bias=0.7):
    print(f"\n===== Checking Spurious-Motif (bias={bias}) =====")

    args = SimpleNamespace()
    args.bias = bias
    args.batch_size = 32

    train_dataset, val_dataset, test_dataset = get_SPMotif(args)

    print("dataset name =", args.dataset)
    print("split sizes:",
          "train =", len(train_dataset),
          "val =", len(val_dataset),
          "test =", len(test_dataset))

    # check one graph
    g0 = train_dataset[0]
    print("example =", g0)
    print("example.x:", None if g0.x is None else tuple(g0.x.shape))
    print("example.edge_index:", tuple(g0.edge_index.shape))
    print("example.y:", g0.y, "shape:", tuple(g0.y.shape))

    # check DataLoader
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    print_batch(batch)

    print("[OK] Spurious-Motif passed.")


if __name__ == "__main__":
    for b in [0.5, 0.7, 0.9]:
        check_spmotif(bias=b)
