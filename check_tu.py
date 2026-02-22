from data_loader_tu import load_tu_dataset, make_loader

def main():
    root = "./data_tu"
    name = "MUTAG"

    print(f"[1] Loading dataset: {name} | root: {root}")
    dataset = load_tu_dataset(name=name, root=root)

    print("[2] Dataset loaded.")
    print("    num_graphs =", len(dataset))
    print("    first graph =", dataset[0])

    loader = make_loader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))

    print("[3] Batch check:")
    print("    batch.x:", tuple(batch.x.shape))
    print("    batch.edge_index:", tuple(batch.edge_index.shape))
    print("    batch.y:", tuple(batch.y.shape))
    print("    batch.batch:", tuple(batch.batch.shape))

    print("[OK] TU dataset pipeline works (PyG official TUDataset).")

if __name__ == "__main__":
    main()
