#!/usr/bin/env python3
"""Assemble a verified Push-T runtime from upstream or an audited local source tree."""

import argparse
import hashlib
import json
from pathlib import Path


RELEASE = Path(__file__).resolve().parent
DATASET = "streaming_flow_policy/pusht/dataset.py"


def digest(content):
    return hashlib.sha256(content).hexdigest()


def fix_state_window_indices(content):
    """Apply the two verified state-dataset edits to the exact audited input."""
    lines = content.splitlines(keepends=True)
    for line_number in (159, 146):
        start = line_number - 1
        lines[start:start + 2] = [
            b"    def __getitem__(self, idx: int):\n",
            b"        datum = super().__getitem__(idx)\n",
        ]
        lines[start + 3:start + 3] = [
            b"            # The GAB table is ordered by this client's dataset windows.\n",
            b'            datum["_win_idx"] = int(idx)\n',
        ]
    return b"".join(lines)


def load_sources(source_root, manifest):
    sources = {}
    for relative in manifest["external_runtime_sources"]:
        path = source_root / relative
        if not path.is_file():
            raise ValueError(f"Missing compatible runtime module: {relative}")
        sources[relative] = path.read_bytes()
    return verify_sources(sources, manifest)


def render_recipe(upstream_lines, recipe):
    chunks = []
    for operation, *args in recipe:
        if operation == "copy":
            source, first, stop = args
            chunks.extend(upstream_lines[source][first:stop])
        elif operation == "indent":
            source, index, indentation = args
            chunks.append(indentation + upstream_lines[source][index].lstrip(" \t"))
        elif operation == "edit":
            source, index, edits = args
            line = upstream_lines[source][index]
            cursor = 0
            for first, stop, replacement in edits:
                chunks.extend((line[cursor:first], replacement))
                cursor = stop
            chunks.append(line[cursor:])
        elif operation == "add":
            chunks.append(args[0])
        else:
            raise ValueError(f"Unknown overlay operation: {operation}")
    return "".join(chunks).encode("utf-8")


def load_upstream_sources(upstream_root, manifest):
    reference = manifest["upstream_reference"]
    sources = {}
    for relative, expected in reference["source_sha256"].items():
        path = upstream_root / relative
        if not path.is_file():
            raise ValueError(f"Missing pinned upstream module: {relative}")
        content = path.read_bytes()
        actual = digest(content)
        if actual != expected:
            raise ValueError(f"Upstream source hash mismatch for {relative}: {actual}")
        sources[relative] = content

    upstream_lines = {
        relative: content.decode("utf-8").splitlines(keepends=True)
        for relative, content in sources.items()
    }
    for relative, recipe_path in reference["edit_recipes"].items():
        recipe = json.loads((RELEASE / recipe_path).read_text(encoding="utf-8"))
        sources[relative] = render_recipe(upstream_lines, recipe)
    for relative, addition_path in reference["standalone_additions"].items():
        sources[relative] = (RELEASE / addition_path).read_bytes()
    return verify_sources(sources, manifest)


def verify_sources(sources, manifest):
    for relative, expected in manifest["external_runtime_sources"].items():
        content = sources[relative]
        actual = digest(content)
        if relative == DATASET and actual == expected:
            content = fix_state_window_indices(content)
            actual = digest(content)
        if relative == DATASET:
            expected = manifest["fixed_dataset_sha256"]
        if actual != expected:
            raise ValueError(f"Source hash mismatch for {relative}: {actual}")
        sources[relative] = content
    for relative, expected in manifest["study_sources"].items():
        content = (RELEASE / relative).read_bytes()
        if digest(content) != expected:
            raise ValueError(f"Packaged study source hash mismatch: {relative}")
        sources[relative] = content
    for relative, content in sources.items():
        compile(content, relative, "exec")
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument("--upstream-root", type=Path, help="Checkout of the pinned upstream revision.")
    origin.add_argument("--source-root", type=Path, help="Existing audited runtime source tree.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--work-dir", type=Path, help="New directory outside the release tree.")
    action.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((RELEASE / "source_requirements.json").read_text(encoding="utf-8"))
    try:
        if args.upstream_root is not None:
            sources = load_upstream_sources(args.upstream_root, manifest)
        else:
            sources = load_sources(args.source_root, manifest)
    except (OSError, ValueError, SyntaxError) as error:
        parser.error(str(error))
    print(f"Verified and syntax-checked {len(sources)} runtime modules.")
    if args.verify_only:
        return

    work_dir = args.work_dir.resolve()
    if work_dir == RELEASE or RELEASE in work_dir.parents:
        parser.error("Keep the locally assembled runtime outside the distributable release tree.")
    try:
        work_dir.mkdir()
    except OSError as error:
        parser.error(f"Cannot create a new runtime directory: {error}")

    for relative, content in sources.items():
        path = work_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    for directory in ("scripts", "scripts/pusht_sfp", "streaming_flow_policy",
                      "streaming_flow_policy/pusht", "streaming_flow_policy/pusht/dp_state_notebook"):
        (work_dir / directory / "__init__.py").write_bytes(b"")
    notice_root = args.upstream_root or args.source_root
    for name in ("LICENSE", "LICENSE.txt", "COPYING", "NOTICE", "NOTICE.txt"):
        notice = notice_root / name
        if notice.is_file():
            (work_dir / name).write_bytes(notice.read_bytes())
    (work_dir / "verified_sources.json").write_text(
        json.dumps({name: digest(content) for name, content in sources.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared local runtime at {work_dir}")


if __name__ == "__main__":
    main()
