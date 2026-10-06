@echo off
rem Sibilance loop, second pass (2026-10-06): the v3 loop plateaued at mix 1.0, but its outputs differed from
rem the v2 plateau by -70..-100 dBFS only, because the guard's fricative detector (hf share >= 0.5) found 12.5%
rem of the harness's fricatives on Vaccin. apl_sibilant_hf_share_min is now a knob (0.2-0.5); start from the v3 final.
cd /d C:\Users\ventu\Projects\AI-Hybrid-VHS-Audio-Restorer
set PYTHONIOENCODING=utf-8
if not exist experiments\autotune_v4 mkdir experiments\autotune_v4
del experiments\autotune_v4\apl_done.txt 2>nul
.venv\Scripts\python.exe scripts\autotune_restoration.py --engine apl --tapes experiments\autotune\tapes_apl.json --out experiments\autotune_v4 --grid scripts\tune_grids\tata_v2.yaml --start-file experiments\autotune_v3\apl\final.json --rounds 20 --parallel 2 >> experiments\autotune_v4\apl_run.log 2>&1
echo DONE > experiments\autotune_v4\apl_done.txt
