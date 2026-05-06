# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from dataclasses import dataclass
from typing import Dict, List

import torch

from cerebras.pytorch.utils.tensorboard import SummaryReader
from cerebras.pytorch.utils.tensorboard.writer import TensorDescriptor


@dataclass
class TensorComparisonResult:
    # Stores tensor names that exists in reference run
    # but missing in CS run.
    missing_cs_names: list = None
    # Stores tensor names that exists in CS run
    # but missing in reference run.
    missing_ref_names: list = None
    # List of tensor names that didn't pass numeric checks.
    numeric_mismatch: list = None
    # List of all other errors happened during comparison.
    unexpected_issues: list = None
    # Total number of tensors checked
    num_comparisons: int = 0

    def __str__(self):
        def append(values, ident=""):
            return f"\n{ident}".join(values)

        messages_list = [f"num_comparisons: {self.num_comparisons}"]

        if len(self.missing_cs_names) != 0:
            missing_cs_names_str = append(self.missing_cs_names, "\t")
            messages_list.append(
                f"missing_cs_names:\n\t{missing_cs_names_str}",
            )

        if len(self.missing_ref_names) != 0:
            missing_ref_names_str = append(self.missing_ref_names, "\t")
            messages_list.append(
                f"missing_ref_names:\n\t{missing_ref_names_str}",
            )

        if len(self.numeric_mismatch) != 0:
            numeric_mismatch_str = append(self.numeric_mismatch, "\t")
            messages_list.append(
                f"numeric_mismatch:\n\t{numeric_mismatch_str}",
            )

        if len(self.unexpected_issues) != 0:
            unexpected_issues_str = append(self.unexpected_issues, "\t")
            messages_list.append(
                f"unexpected comparison errors:\n\t{unexpected_issues_str}",
            )

        return append(messages_list)

    def failed(self):
        return (
            len(self.missing_cs_names) != 0
            or len(self.missing_ref_names) != 0
            or len(self.numeric_mismatch) != 0
            or len(self.unexpected_issues) != 0
            or not self.num_comparisons
        )


def compare_log_dirs(
    cs_log_dir, ref_log_dir, rtol: float
) -> TensorComparisonResult:
    cs_reader = SummaryReader(cs_log_dir)
    ref_reader = SummaryReader(ref_log_dir)

    cs_tensor_names = set(cs_reader.tensor_names())
    ref_tensor_names = set(ref_reader.tensor_names())

    res = TensorComparisonResult(
        missing_cs_names=list(ref_tensor_names - cs_tensor_names),
        missing_ref_names=list(cs_tensor_names - ref_tensor_names),
        numeric_mismatch=list(),
        unexpected_issues=list(),
    )

    tensor_names = cs_tensor_names & ref_tensor_names
    for tensor_name in tensor_names:
        cs_tensors: Dict[int, List[TensorDescriptor]] = cs_reader.read_tensor(
            tensor_name, latest_only=False
        )
        ref_tensors: Dict[int, List[TensorDescriptor]] = ref_reader.read_tensor(
            tensor_name, latest_only=False
        )

        # Check for missing steps.
        cs_steps = set(cs_tensors.keys())
        ref_steps = set(ref_tensors.keys())

        missing_cs_steps = ref_steps - cs_steps
        missing_ref_steps = cs_steps - ref_steps

        for step in missing_cs_steps:
            res.missing_cs_names.append(f"{tensor_name}.{step}")

        for step in missing_ref_steps:
            res.missing_ref_names.append(f"{tensor_name}.{step}")

        # Compare numerics.
        steps = ref_steps & cs_steps

        for step in steps:
            if len(cs_tensors[step]) != len(ref_tensors[step]):
                res.unexpected_issues.append(
                    f"Mismatching number of tensor values ({len(cs_tensors[step])} vs. "
                    f"{len(ref_tensors[step])}) found for tensor with name "
                    f"\"{tensor_name}\" at step {step}."
                )
                continue

            for idx, (cs_tensor, ref_tensor) in enumerate(
                zip(cs_tensors[step], ref_tensors[step])
            ):
                cs_tensor = cs_tensor.tensor.to(torch.float32)
                ref_tensor = ref_tensor.tensor.to(torch.float32)

                res.num_comparisons += 1
                if not torch.allclose(cs_tensor, ref_tensor, rtol=rtol):
                    res.numeric_mismatch.append(f"{tensor_name}.{step}[{idx}]")

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cs_log_dir", required=True, type=str)
    parser.add_argument("--ref_log_dir", required=True, type=str)
    parser.add_argument("--rtol", default=0.05, type=float)

    args = parser.parse_args()

    res = compare_log_dirs(args.cs_log_dir, args.ref_log_dir, args.rtol)
    if res.failed():
        print(f"Comparison failed:\n{res}")
    else:
        print(f"Comparison successful!")
