import os
import torch
import opts
import utils
from torch_geometric.loader import DataLoader


def _print_batch(batch):
    print("batch.x:", None if batch.x is None else tuple(batch.x.shape))
    print("batch.edge_index:", tuple(batch.edge_index.shape))
    print("batch.y:", None if batch.y is None else tuple(batch.y.shape))
    print("batch.batch:", tuple(batch.batch.shape))


def check_syn(bias=0.7, data_root="./data_syn_check", data_num=200, batch_size=32):
    """
    Check ICL synthetic pipeline (main_syn.py):
      - load or generate syn_dataset.pt via utils.graph_dataset_generate
      - split via utils.dataset_bias_split
      - build DataLoader and fetch one batch

    bias: 0.5 / 0.7 / 0.9 (or any float used by ICL)
    data_root: cache directory for syn_dataset.pt
    data_num: controls total size (ICL uses total = data_num * 4)
    batch_size: DataLoader batch size for checking
    """
    print(f"\n===== Checking SYN (main_syn pipeline) | bias={bias} =====")

    # ✅ Get full args from repo defaults to avoid missing fields (node_num, etc.)
    args = opts.parse_args()

    # ✅ Override only what we need for check
    args.hidden = 32
    args.bias = bias
    args.data_root = data_root
    args.data_num = data_num
    args.batch_size = batch_size

    os.makedirs(args.data_root, exist_ok=True)
    save_path = args.data_root
    cache_path = os.path.join(save_path, "syn_dataset.pt")

    # Load or generate dataset (same logic as main_syn.py)
    try:
        if os.path.exists(cache_path):
            print("[1] Loading cached syn_dataset.pt ...")
            dataset = torch.load(cache_path)
        else:
            raise FileNotFoundError
    except Exception:
        print("[1] Generating syn dataset ...")
        dataset = utils.graph_dataset_generate(args, save_path)
        # In case generate() doesn't save by itself, we save it
        if not os.path.exists(cache_path):
            torch.save(dataset, cache_path)

    # Split (same as main_syn.py)
    train_set, val_set, test_set, the = utils.dataset_bias_split(
        dataset, args, bias=args.bias, split=[7, 1, 2], total=args.data_num * 4
    )

    print("[2] Split sizes:",
          "train =", len(train_set),
          "val =", len(val_set),
          "test =", len(test_set))

    # Optional: print dataset info (if util supports)
    try:
        _ = utils.print_dataset_info(train_set, val_set, test_set, the)
    except Exception as e:
        print("[WARN] print_dataset_info failed:", repr(e))

    # Check one graph
    g0 = train_set[0]
    print("[3] Example =", g0)

    # DataLoader check
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    batch = next(iter(loader))
    print("[4] Batch check:")
    _print_batch(batch)

    print("[OK] SYN pipeline works (generated + bias split + loader).")


if __name__ == "__main__":
    # ✅ Start with one bias to reduce generation time; you can add more later.
    check_syn(bias=0.7, data_root="./data_syn_check", data_num=200, batch_size=32)

    # If you want to test all three biases, uncomment:
    # for b in [0.5, 0.7, 0.9]:
    #     check_syn(bias=b, data_root=f"./data_syn_check_b{b}", data_num=200, batch_size=32)
