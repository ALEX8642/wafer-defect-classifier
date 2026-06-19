"""
evaluate.py — Test-set evaluation: confusion matrix, per-class metrics, macro-F1.
Phase 1: implement evaluate(cfg, checkpoint_path).

Entry point: python -m wafer.evaluate --config configs/baseline.yaml --checkpoint <path>
"""
# Phase 1


def evaluate(cfg, checkpoint_path=None):
    raise NotImplementedError("Phase 1: implement evaluate.py")


if __name__ == "__main__":
    from wafer.config import WaferConfig, build_arg_parser

    parser = build_arg_parser("wafer evaluate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    args = parser.parse_args()
    cfg = WaferConfig.from_yaml_and_args(args.config, args)
    evaluate(cfg, checkpoint_path=args.checkpoint)
