import os
import torch
from types import SimpleNamespace
from torch_geometric.loader import DataLoader
import utils

def _print_batch(batch):
    print("batch.x:", None if batch.x is None else tuple(batch.x.shape))
    print("batch.edge_index:", tuple(batch.edge_index.shape))
    print("batch.y:", None if batch.y is None else tuple(batch.y.shape))
    print("batch.batch:", tuple(batch.batch.shape))

def check_syn(bias=0.7, data_root="./data_syn_check", data_num=100):
    """
    bias: synthetic bias level (follow ICL args)
    data_root: where syn_dataset.pt will be cached
    data_num: controls total size (ICL uses total = data_num * 4)
    """
    print(f"\n===== Checking SYN (main_syn pipeline) | bias={bias} =====")

    os.makedirs(data_root, exist_ok=True)

    # mimic args needed by utils.graph_dataset_generate / dataset_bias_split
    args = SimpleNamespace()
    args.data_root = data_root
    args.bias = bias
    args.data_num = data_num
    args.batch_size = 32

    save_path = data_root
    cache_path = os.path.join(save_path, "syn_dataset.pt")

    # load or generate
    if os.path.exists(cache_path):
        print("[1] Loading cached syn_dataset.pt ...")
        dataset = torch.load(cache_path)
    else:
        print("[1] Generating syn dataset ...")
        dataset = utils.graph_dataset_generate(args, save_path)
        # main_syn does torch.save inside generate or after; but we keep it safe:
        if not os.path.exists(cache_path):
            torch.save(dataset, cache_path)

    # split (same as main_syn)
    train_set, val_set, test_set, the = utils.dataset_bias_split(
        dataset, args, bias=args.bias, split=[7, 1, 2], total=args.data_num * 4
    )

    print("[2] Split sizes:",
          "train =", len(train_set),
          "val =", len(val_set),
          "test =", len(test_set))

    # print dataset info if available
    try:
        _ = utils.print_dataset_info(train_set, val_set, test_set, the)
    except Exception as e:
        print("[WARN] print_dataset_info failed:", repr(e))

    # basic graph check
    g0 = train_set[0]
    print("[3] Example =", g0)

    # loader check
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    batch = next(iter(loader))
    print("[4] Batch check:")
    _print_batch(batch)

    print("[OK] SYN pipeline works (generated + bias split + loader).")

if __name__ == "__main__":
    # You can test multiple bias values if you want:
    for b in [0.5, 0.7, 0.9]:
        check_syn(bias=b, data_root="./data_syn_check", data_num=200)
