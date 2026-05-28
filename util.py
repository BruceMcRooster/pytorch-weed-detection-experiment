import argparse
import inspect

# total AI slop, hoping to save some typing
def add_subparser(subparsers, func, command_name: str):
    """Dynamically creates a subcommand parser from a function signature."""
    # Create the specific subcommand parser
    subparser = subparsers.add_parser(command_name, help=func.__doc__)

    # Inspect the function signature
    sig = inspect.signature(func)

    for name, param in sig.parameters.items():
        # Handle arguments with defaults (options/flags)
        if param.default != inspect.Parameter.empty:
            if isinstance(param.default, bool):
                subparser.add_argument(f"--{name}", action="store_true", help=f"(Default: {param.default})")
            else:
                subparser.add_argument(f"--{name}", type=type(param.default), default=param.default, help=f"(Default: {param.default})")
        # Handle required positional arguments
        else:
            subparser.add_argument(name, type=str, help="Required parameter")

    # Save the original function link inside the parser options
    subparser.set_defaults(func=func)