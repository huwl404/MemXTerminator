from __future__ import annotations


def main() -> None:
    from memxterminator.mxt_state import (
        to_output_micrograph_path,
        to_output_stack_path,
        to_output_stack_path_in_root,
        to_subtracted_micrograph_path,
        to_subtracted_stack_path,
        to_subtracted_stack_path_in_root,
        validate_output_dirname,
    )

    # Particle stack mapping (legacy base behavior).
    assert to_subtracted_stack_path("/a/extract/x.mrc") == "/a/subtracted/x_subtracted.mrc"
    assert to_subtracted_stack_path("/a/subtracted/x_subtracted.mrc") == "/a/subtracted/x_subtracted.mrc"
    assert to_output_stack_path("/a/extract/x.mrc", output_dirname="class_01") == "/a/class_01/x_subtracted.mrc"
    assert to_output_stack_path("/a/extract/空 格.mrc", output_dirname="输出目录") == "/a/输出目录/空 格_subtracted.mrc"

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
    assert (
        to_output_stack_path_in_root("/a/extract/x.mrc", output_root="/tmp/run1", output_dirname="class_02")
        == "/tmp/run1/class_02/x_subtracted.mrc"
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
    assert (
        to_output_micrograph_path("/a/micrographs/extract/mg_001.mrc", output_root="/tmp/run2", output_dirname="class_02")
        == "/tmp/run2/class_02/mg_001_subtracted.mrc"
    )

    assert validate_output_dirname("subtracted") == "subtracted"
    for bad in ("", ".", "..", "a/b", "a\\b"):
        try:
            validate_output_dirname(bad)
            raise AssertionError(f"Expected validate_output_dirname to fail for {bad!r}")
        except ValueError:
            pass

    print(">>> OK: output_root mapping helpers.")


if __name__ == "__main__":
    main()
