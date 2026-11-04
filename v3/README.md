- run_pipeline_LEGGOS_v3.py:  a working script that reduces LEGGOS JWST NIRSpec IFU data.  Originally a script by Brian Welch that ran v1.20 of the pipeline.  Jane Rigby upgraded to run v2 and v3 of the pipeline, to check for missing CRDS files, and avoid collisions when run in parallel.  Codex 5.6 was used as a coding assistant.  See prompt files for how I did that.

- run_all_IFS_targets.py: A little wrapper that calls run_pipeline_LEGGOS_v3.py for all the LEGGOS targets, and parallelizes for speed.  
