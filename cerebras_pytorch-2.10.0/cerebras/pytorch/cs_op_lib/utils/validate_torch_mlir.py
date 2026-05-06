#!/usr/bin/env python3
# Copyright 2016-2024 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

"""
Traverse a Torch MLIR file for Cerebras constraint validation.
"""

import argparse
import logging
import sys
from typing import Dict, Optional

import cerebras.pytorch.lib._torch_mlir as torch_mlir
from cerebras.pytorch.cs_op_lib.errors import (
    ConstraintValidationError,
    OpNotFoundError,
)
from cerebras.pytorch.cs_op_lib.ops.registry import (
    CsOpRegistry,
    register_all_ops,
)
from cerebras.pytorch.lib._torch_mlir.dialects import torch as torch_d

LOGGER = logging.getLogger("validate_torch_mlir")


def traverse_ops(
    op: torch_mlir.ir.Operation, allow_undefined_op: bool = True
) -> Dict[str, str]:
    """
    Recursively traverse operations, logging Torch ops' names and
    checking constraint validity against Cerebras ops.

    Args:
        op: The current operation to traverse (MLIR operation).
        allow_undefined_op: If True, suppress errors for ops not found in the
                            Cerebras Op Library. Default is True.
    Returns:
        Dict[str, str]: A dictionary of op names that failed validation and the reason.
    """
    failed_ops: Dict[str, str] = {}

    # Convert from OpView to operation if necessary
    if isinstance(op, torch_mlir._mlir_libs._mlir.ir.OpView):
        op = op.operation

    # If this is a torch.aten op, check constraints
    if op.name.startswith("torch.aten"):
        op_name = op.name[len("torch.aten.") :]  # Remove the prefix
        try:
            cs_op = CsOpRegistry.get_op_class(op_name)
        except OpNotFoundError as e:
            if allow_undefined_op:
                LOGGER.warning(e)
            else:
                raise
        else:
            try:
                cs_op.from_mlir(op.operands).validate()
            except ConstraintValidationError as e:
                LOGGER.error(f"'{op_name}': {e}")
                failed_ops[op_name] = str(e)
            else:
                LOGGER.info(f"'{op_name}' passed constraint validation.")

    # Recursively visit each region, block, and nested operation
    for region in op.regions:
        for block in region.blocks:
            for inner_op in block.operations:
                failed_ops.update(
                    traverse_ops(
                        inner_op, allow_undefined_op=allow_undefined_op
                    )
                )

    return failed_ops


def convert_mlir_version(mlir_text: str) -> str:
    """
    Newer versions of Torch-MLIR expects a func.func as the top-level
    operation. This function converts the MLIR text to ensure it has
    a `func.func` as the top-level operation.
    Args:
        mlir_text (str): MLIR text string containing Torch ops.
    Returns:
        str: Converted MLIR text with a `func.func` as the top-level operation.
    """

    # Check if the top-level operation is already a func.func
    if "func.func" in mlir_text:
        return mlir_text
    # If not, wrap the existing MLIR text in a func.func
    else:
        return mlir_text.replace(
            "func", "func.func", 1
        )  # Replace the first occurrence of "func" with "func.func"


def check_mlir_text(
    mlir_text: str, definitions_module: Optional[str] = None
) -> Dict[str, str]:
    """
    Check constraints of Torch ops in an MLIR text string. This function
    can be used as a library function.

    Args:
        mlir_text (str): MLIR text string containing Torch ops.
    Raises:
        RuntimeError: If we fail to parse the MLIR.
    Returns:
        Dict[str, str]: A dictionary of op names that failed validation and the reason.
    """

    mlir_text = convert_mlir_version(mlir_text)

    # Register all ops (safe to call multiple times in practice, but
    # normally done once per process)
    register_all_ops(definitions_module)

    # Use a Torch-MLIR context/dialect
    with torch_mlir.ir.Context() as ctx:
        torch_d.register_dialect(ctx)
        LOGGER.debug("Parsing MLIR file...")
        try:
            module = torch_mlir.ir.Module.parse(mlir_text)
        except Exception as e:
            raise RuntimeError(f"Failed to parse MLIR input: {e}") from e

        # The top-level MLIR object in Python is a `Module`.
        top_level_op = module.operation

        # Recursively traverse and validate Torch ops
        LOGGER.debug("Starting operation traversal...")
        failed_ops = traverse_ops(top_level_op)
        LOGGER.debug("Traversal complete.")
        return failed_ops


def check_mlir_file(
    mlir_file_path: str, definitions_module: Optional[str] = None
) -> Dict[str, str]:
    """
    Check constraints of Torch ops in an MLIR file. Can be used as a library
    function without configuring logging or exiting the process.

    Args:
        mlir_file_path (str): Path to the MLIR file.

    Raises:
        FileNotFoundError: If the file is not found.
        IOError: If there are issues reading the file.
        RuntimeError: If we fail to parse the MLIR.
    Returns:
        Dict[str, str]: A dictionary of op names that failed validation and the reason.
    """
    LOGGER.debug(f"Reading MLIR file from: {mlir_file_path}")

    with open(mlir_file_path, "r", encoding="utf-8") as f:
        mlir_input = f.read()

    return check_mlir_text(mlir_input, definitions_module)


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Traverse a Torch MLIR file for Cerebras constraint validation."
    )
    parser.add_argument(
        "mlir_file_path",
        type=str,
        help="Path to the MLIR file containing Torch ops.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level. Default is INFO.",
    )
    parser.add_argument(
        "--definitions-module",
        type=str,
        default=None,
        help="Module containing the definitions of ops to register. If not provided, uses the default definitions module.",
    )
    return parser.parse_args()


def main():
    """
    Main entry point for the script. Sets up logging, registers ops,
    loads and parses the MLIR file, and traverses operations.
    """
    args = parse_args()

    # Configure logging for script usage
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        stream=sys.stdout,
    )

    try:
        failed_ops = check_mlir_file(
            args.mlir_file_path, args.definitions_module
        )
    except FileNotFoundError:
        LOGGER.error(f"File not found: {args.mlir_file_path}")
        sys.exit(1)
    except IOError as e:
        LOGGER.error(f"I/O error reading file '{args.mlir_file_path}': {e}")
        sys.exit(1)
    except RuntimeError as e:
        LOGGER.error(e)
        sys.exit(1)
    except OpNotFoundError as e:
        LOGGER.error(e)
        sys.exit(1)
    except Exception as e:
        # Catch any other unforeseen exceptions
        LOGGER.exception(f"Unexpected error: {e}")
        sys.exit(1)
    else:
        if failed_ops:
            LOGGER.error(
                "One or more constraint validation errors occurred. Exiting with error."
            )
            LOGGER.info("--- Constraint Validation Report ---")
            for op_name, reason in failed_ops.items():
                LOGGER.info(f"Op: {op_name}\nReason: {reason}\n")
            LOGGER.info("--- End of Report ---")
            sys.exit(1)

    LOGGER.info("All Torch ops passed constraint validation.")


if __name__ == "__main__":
    main()
