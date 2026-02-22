import torch
from torch_geometric.loader import DataLoader
from TU_dataset import TUDatasetExt

def main():
    root = "./data_tu"   # 可以换成别的路径
    name = "MUTAG"     

    print(f"[1] Loading dataset: {name} | root: {root}")
    dataset = TUDatasetExt(root=root, name=name)

    print("[2] Dataset loaded.")
    print("    num_graphs =", len(dataset))
    print("    first graph =", dataset[0])

    # 检查字段
    g0 = dataset[0]
    print("[3] Field check:")
    print("    x:", None if g0.x is None else tuple(g0.x.shape))
    print("    edge_index:", tuple(g0.edge_index.shape))
    print("    y:", g0.y, "shape:", tuple(g0.y.shape))

    # DataLoader 检查 batch
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))

    print("[4] Batch check:")
    print("    batch.x:", None if batch.x is None else tuple(batch.x.shape))
    print("    batch.edge_index:", tuple(batch.edge_index.shape))
    print("    batch.y:", tuple(batch.y.shape))
    print("    batch.batch:", tuple(batch.batch.shape))

    print("[OK] TU dataset pipeline works.")

if __name__ == "__main__":
    main()
