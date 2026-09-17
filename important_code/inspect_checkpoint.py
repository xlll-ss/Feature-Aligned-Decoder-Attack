import argparse

import torch

from train_decoder_multidataset import extract_state_dict, infer_checkpoint_num_classes, strip_state_prefix


def main():
    parser = argparse.ArgumentParser(description="Inspect a target or decoder checkpoint without loading a model")
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = extract_state_dict(checkpoint)
    print(f"checkpoint: {args.checkpoint}")
    if isinstance(checkpoint, dict):
        for key in ("dataset", "arch", "epoch", "num_classes", "feature_dim", "img_size", "val_acc"):
            if key in checkpoint:
                print(f"{key}: {checkpoint[key]}")
    print(f"inferred_num_classes: {infer_checkpoint_num_classes(state)}")
    for key, value in state.items():
        clean_key = strip_state_prefix(key)
        if clean_key in {"fc_layer.weight", "fc_layer.bias", "feat_proj.weight", "bn.weight"}:
            print(f"{clean_key}: {tuple(value.shape)}")


if __name__ == "__main__":
    main()
