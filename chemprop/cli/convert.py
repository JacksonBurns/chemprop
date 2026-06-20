from argparse import ArgumentError, ArgumentParser, Namespace
import logging
from pathlib import Path
import sys

from chemprop.cli.utils import Subcommand
from chemprop.utils.v1_to_v2 import convert_model_file_v1_to_v2
from chemprop.utils.v2_0_to_v2_1 import convert_model_file_v2_0_to_v2_1

logger = logging.getLogger(__name__)


class ConvertSubcommand(Subcommand):
    COMMAND = "convert"
    HELP = "Convert model checkpoint (.pt) to more recent version or to PyTorch Geometric format."

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "-c",
            "--conversion",
            choices=["v1_to_v2", "v2_0_to_v2_1", "to_pyg"],
            help=(
                "Conversion to perform. Models converted from v1 to v2 must be run with the v1 "
                "featurizer via `--multi-hot-atom-featurizer-mode v1`. Use `to_pyg` to convert "
                "a v2 model to PyTorch Geometric format (requires `pip install chemprop[pyg]`)."
            ),
            default="v1_to_v2",
        )
        parser.add_argument(
            "-i",
            "--input-path",
            required=True,
            type=Path,
            help="Path to a model .pt checkpoint file",
        )
        parser.add_argument(
            "-o",
            "--output-path",
            type=Path,
            help="Path to which the converted model will be saved (``CURRENT_DIRECTORY/STEM_OF_INPUT_newversion.pt`` by default)",
        )
        return parser

    @classmethod
    def func(cls, args: Namespace):
        _suffix = None
        if args.conversion == "v1_to_v2":
            _suffix = "_v2.pt"
        elif args.conversion == "v2_0_to_v2_1":
            _suffix = "_v2_1.pt"
        elif args.conversion == "to_pyg":
            _suffix = "_pyg.pt"
        if args.output_path is None:
            args.output_path = Path(args.input_path.stem + _suffix)
        if args.output_path.suffix != ".pt":
            raise ArgumentError(
                argument=None, message=f"Output must be a `.pt` file. Got {args.output_path}"
            )

        logger.info(
            f"Converting model checkpoint '{args.input_path}' to model checkpoint '{args.output_path}'..."
        )
        if args.conversion == "v1_to_v2":
            convert_model_file_v1_to_v2(args.input_path, args.output_path)
        elif args.conversion == "v2_0_to_v2_1":
            convert_model_file_v2_0_to_v2_1(args.input_path, args.output_path)
        elif args.conversion == "to_pyg":
            try:
                from chemprop.utils.convert_to_pyg import convert_model_file_to_pyg
            except ImportError:
                raise ImportError(
                    "Converting to PyTorch Geometric format requires the 'pyg' optional dependency. "
                    "Install it with: pip install chemprop[pyg]"
                )
            convert_model_file_to_pyg(args.input_path, args.output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = ConvertSubcommand.add_args(parser)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

    args = parser.parse_args()
    ConvertSubcommand.func(args)
