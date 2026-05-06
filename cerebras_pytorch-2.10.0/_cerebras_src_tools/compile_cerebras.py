from __future__ import annotations

import argparse
import importlib.util
import py_compile
import shutil
import sys
from pathlib import Path


def compile_tree(source_dir: Path, output_dir: Path, clean: bool) -> int:
    if not source_dir.is_dir():
        print(f"Source directory does not exist: {source_dir}", file=sys.stderr)
        return 2

    project_root = source_dir.parent
    unsafe_outputs = {project_root, source_dir}
    if output_dir in unsafe_outputs or source_dir in output_dir.parents:
        print(f"Refusing to clean unsafe output directory: {output_dir}", file=sys.stderr)
        return 2

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)

    failures = []
    compiled_count = 0
    for source_file in sorted(source_dir.rglob("*.py")):
        relative_source = source_file.relative_to(source_dir.parent)
        output_source = output_dir / relative_source
        compiled_file = Path(importlib.util.cache_from_source(str(output_source)))
        compiled_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            py_compile.compile(
                str(source_file),
                cfile=str(compiled_file),
                dfile=str(relative_source),
                doraise=True,
                invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH,
            )
            compiled_count += 1
        except py_compile.PyCompileError as error:
            failures.append((relative_source, error.msg))

    if failures:
        for relative_source, message in failures:
            print(f"{relative_source}: {message}", file=sys.stderr)
        return 1

    print(f"Compiled {compiled_count} files to {output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Compile Python files under the unpacked cerebras directory."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "cerebras",
        help="Source directory to compile.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "build" / "pyc",
        help="Directory where compiled bytecode should be written.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep any existing compiled bytecode before compiling.",
    )
    args = parser.parse_args(argv)

    return compile_tree(
        args.source.resolve(),
        args.output.resolve(),
        clean=not args.no_clean,
    )


if __name__ == "__main__":
    raise SystemExit(main())
