"""
train.py — Training loop: AdamW, cosine LR, early stopping on val macro-F1.
Phase 1: implement train(cfg).

Entry point: python -m wafer.train --config configs/baseline.yaml
"""
# Phase 1


def train(cfg):
    raise NotImplementedError("Phase 1: implement train.py")


if __name__ == "__main__":
    from wafer.config import WaferConfig, build_arg_parser

    parser = build_arg_parser("wafer train")
    args = parser.parse_args()
    cfg = WaferConfig.from_yaml_and_args(args.config, args)
    train(cfg)
