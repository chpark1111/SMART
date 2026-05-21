import os
import pathlib
import sys

_SMART_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PACKAGED_MANIFOLD_PYTHON = _SMART_ROOT / "pymanifold_runtime"
_SOURCE_MANIFOLD_PYTHON = _SMART_ROOT / "vendor" / "manifold" / "build" / "bindings" / "python"
_DEFAULT_MANIFOLD_PYTHON = (
    _PACKAGED_MANIFOLD_PYTHON
    if any(_PACKAGED_MANIFOLD_PYTHON.glob("pymanifold*"))
    else _SOURCE_MANIFOLD_PYTHON
)
_MANIFOLD_PYTHON = pathlib.Path(
    os.environ.get("SMART_MANIFOLD_PYTHON", str(_DEFAULT_MANIFOLD_PYTHON))
)
if _MANIFOLD_PYTHON.exists():
    sys.path.insert(0, str(_MANIFOLD_PYTHON))

from configs.args import get_parser


def run() -> None:
    parser = get_parser()
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.run_type not in {"greedy", "test_env", "mcts"}:
        import torch

        if torch.cuda.is_available():
            print("Using GPU %s" % (args.gpu))

    if args.run_type == "train":
        from src.train import train

        train(args)
    elif args.run_type == "mcts":
        from src.mcts import mcts

        mcts(args)
    elif args.run_type == "greedy":
        from src.greedy import greedy

        greedy(args)
    elif args.run_type == "test_env":
        from tests.environment.test_environment import test_env

        test_env(args)
    elif args.run_type == "eval":
        from src.eval import eval

        checkpoint_dir = (
            os.environ.get("SMART_REFINE_CHECKPOINT_DIR", "./logs/exp/")
        )
        eval(
            parser=parser,
            exp_path=os.path.join(checkpoint_dir, args.path_to_pt_file),
            pt_id=args.load_idx,
            debug=args.debug,
            final_k=args.final_k,
        )
    else:
        print('"%s"is not a possible running type' % (args.run_type))


if __name__ == "__main__":
    run()
