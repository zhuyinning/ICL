import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader

DIR_DATASETS = [
    "ogbg-molhiv",
    # you can add more OGB graph property prediction datasets here if needed:
    # "ogbg-molpcba",
    # "ogbg-moltox21",
    # "ogbg-molbace",
    # "ogbg-molbbbp",
]

def verify_dataset(name: str, root: str = "./data_dir"):
    print(f"\n===== Checking {name} =====")

    try:
        dataset = PygGraphPropPredDataset(name=name, root=root)

        print("num_graphs =", len(dataset))
        print("num_tasks  =", dataset.num_tasks)
        print("example    =", dataset[0])

        # Official split check
        split_idx = dataset.get_idx_split()
        print(
            "split sizes:",
            "train =", len(split_idx["train"]),
            "valid =", len(split_idx["valid"]),
            "test  =", len(split_idx["test"]),
        )

        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        batch = next(iter(loader))

        print("batch.x:", None if batch.x is None else tuple(batch.x.shape))
        print("batch.edge_index:", tuple(batch.edge_index.shape))
        print("batch.y:", None if batch.y is None else tuple(batch.y.shape))
        print("batch.batch:", tuple(batch.batch.shape))

        print(f"[OK] {name}")

    except Exception as e:
        print(f"[FAILED] {name}: {repr(e)}")


def main():
    for name in DIR_DATASETS:
        verify_dataset(name)


if __name__ == "__main__":
    main()
