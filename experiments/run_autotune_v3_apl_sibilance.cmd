@echo off
rem Listening round 2 (2026-10-05): the user preferred APL on speech, heard natural pauses and fine music,
rem and still heard APL's 's' thin. APL on the Tata tapes from the shipped defaults (the v2 final) with the
rem sibilant guard's mix up to 1.0 and its crossover (2-5 kHz) as knobs; stops only at the plateau.
cd /d C:\Users\ventu\Projects\AI-Hybrid-VHS-Audio-Restorer
set PYTHONIOENCODING=utf-8
if not exist experiments\autotune_v3 mkdir experiments\autotune_v3
del experiments\autotune_v3\apl_done.txt 2>nul
.venv\Scripts\python.exe scripts\autotune_restoration.py --engine apl --tapes experiments\autotune\tapes_apl.json --out experiments\autotune_v3 --grid scripts\tune_grids\tata_v2.yaml --start-file experiments\autotune_v2\apl\final.json --rounds 20 --parallel 2 >> experiments\autotune_v3\apl_run.log 2>&1
echo DONE > experiments\autotune_v3\apl_done.txt
