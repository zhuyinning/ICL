from data_loader_tu import load_tu_dataset, make_loader

TU_DATASETS = [
    "MUTAG",
    "NCI1",
    "PROTEINS",
    "IMDB-BINARY",
    "IMDB-MULTI",
    "COLLAB",
]

def verify_dataset(name, root="./data_tu"):
    print(f"\n===== Checking {name} =====")

    try:
        dataset = load_tu_dataset(name=name, root=root)
        print(f"num_graphs = {len(dataset)}")
        print(f"example = {dataset[0]}")

        loader = make_loader(dataset, batch_size=32, shuffle=True)
        batch = next(iter(loader))

        print("batch.x:", tuple(batch.x.shape))
        print("batch.edge_index:", tuple(batch.edge_index.shape))
        print("batch.y:", tuple(batch.y.shape))
        print("batch.batch:", tuple(batch.batch.shape))

        print(f"[OK] {name}")

    except Exception as e:
        print(f"[FAILED] {name}: {repr(e)}")

def main():
    for name in TU_DATASETS:
        verify_dataset(name)

if __name__ == "__main__":
    main()
