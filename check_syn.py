import os
import torch
import opts
import utils
from torch_geometric.loader import DataLoader


def _print_batch(batch):
    print("batch.x:", None if getattr(batch, "x", None) is None else tuple(batch.x.shape))
    print("batch.edge_index:", tuple(batch.edge_index.shape))
    print("batch.y:", None if getattr(batch, "y", None) is None else tuple(batch.y.shape))
    print("batch.batch:", tuple(batch.batch.shape))


def check_syn(bias=0.7, base_root="./data_syn_check", data_num=200, batch_size=32):
    print(f"\n==============================")
    print(f"Checking SYN (bias={bias})")
    print(f"==============================")

    # 获取完整 args
    args = opts.parse_args()

    args.hidden = 32
    args.bias = bias
    args.data_root = os.path.join(base_root, f"bias_{bias}")
    args.data_num = data_num
    args.batch_size = batch_size

    os.makedirs(args.data_root, exist_ok=True)
    cache_path = os.path.join(args.data_root, "syn_dataset.pt")
    
    if os.path.exists(cache_path):
        print("[1] Loading cached syn_dataset.pt ...")
        dataset = torch.load(cache_path)
    else:
        print("[1] Generating syn dataset ...")
        dataset = utils.graph_dataset_generate(args, args.data_root)
        torch.save(dataset, cache_path)
        print(f"[1] Saved at: {cache_path}")

    # bias split
    train_set, val_set, test_set, the = utils.dataset_bias_split(
        dataset,
        args,
        bias=args.bias,
        split=[7, 1, 2],
        total=args.data_num * 4
    )

    print("[2] Split sizes:",
          "train =", len(train_set),
          "val =", len(val_set),
          "test =", len(test_set))

    try:
        utils.print_dataset_info(train_set, val_set, test_set, the)
    except:
        pass

    g0 = train_set[0]
    print("[3] Example graph:", g0)

    # DataLoader 测试
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    batch = next(iter(loader))

    print("[4] Batch check:")
    _print_batch(batch)

    print(f"[OK] bias={bias} passed.\n")


if __name__ == "__main__":

    print("############################################")
    print("Running SYN check for all bias values")
    print("############################################")

    for b in [0.5, 0.7, 0.9]:
        check_syn(
            bias=b,
            base_root="./data_syn_check",
            data_num=200,
            batch_size=32
        )

    print("############################################")
    print("All SYN bias checks completed.")
    print("############################################")
