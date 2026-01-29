from __future__ import annotations


def main() -> None:
    from memxterminator.mxt_state import (
        to_subtracted_micrograph_path,
        to_subtracted_stack_path,
        to_subtracted_stack_path_in_root,
    )

    # Particle stack mapping (legacy base behavior).
    assert to_subtracted_stack_path("/a/extract/x.mrc") == "/a/subtracted/x_subtracted.mrc"
    assert to_subtracted_stack_path("/a/subtracted/x_subtracted.mrc") == "/a/subtracted/x_subtracted.mrc"

    # Particle stack mapping with output_root override.
    assert (
        to_subtracted_stack_path_in_root("/a/extract/x.mrc", output_root="/tmp/run1")
        == "/tmp/run1/subtracted/x_subtracted.mrc"
    )
    # Even if the input path already contains 'subtracted', the output_root should isolate outputs.
    assert (
        to_subtracted_stack_path_in_root("/a/subtracted/x_subtracted.mrc", output_root="/tmp/run1")
        == "/tmp/run1/subtracted/x_subtracted.mrc"
    )

    # Micrograph mapping (default behavior).
    assert (
        to_subtracted_micrograph_path("/a/micrographs/extract/mg_001.mrc")
        == "/a/micrographs/subtracted/mg_001_subtracted.mrc"
    )
    # Micrograph mapping with output_root override.
    assert (
        to_subtracted_micrograph_path("/a/micrographs/extract/mg_001.mrc", output_root="/tmp/run2")
        == "/tmp/run2/subtracted/mg_001_subtracted.mrc"
    )

    print(">>> OK: output_root mapping helpers.")


if __name__ == "__main__":
    main()

