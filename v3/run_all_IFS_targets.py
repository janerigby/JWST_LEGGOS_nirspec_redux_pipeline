#!/usr/bin/env python
"""Run the NIRSpec IFU pipeline for every target directory in Raw_data."""

import glob
import os
import time

import run_pipeline_LEGGOS_v3 as pipeline


CORES_TO_USE = 4


def main():
    start_time = time.monotonic()

    target_directories = sorted(
        entry.name
        for entry in os.scandir(pipeline.dir_inputdata)
        if entry.is_dir()
        and glob.glob(os.path.join(entry.path, "**", "*_uncal.fits"), recursive=True)
    )

    if not target_directories:
        raise RuntimeError(
            f"No target directories containing *_uncal.fits were found in "
            f"{pipeline.dir_inputdata}"
        )

    print("Targets to process:", ", ".join(target_directories))

    for target in target_directories:
        print(f"\n{'=' * 70}\nStarting {target}\n{'=' * 70}")
        pipeline.run_pipeline(
            target,
            run_bgsub=False,
            cube_resolutions=(0.1,),
            cores2use=CORES_TO_USE,
        )

    elapsed_seconds = time.monotonic() - start_time
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(
        f"\nCompleted all targets in "
        f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} (HH:MM:SS)."
    )


if __name__ == "__main__":
    main()
