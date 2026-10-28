@echo off
rem Generated launcher. Logs go to data\*.log, progress lines to data\status.txt.
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set CF_TAG=_v2
echo [%date% %time%] v2 start >> data\status.txt
echo [%date% %time%] v2: fine-tune n s m >> data\status.txt
%PY% 03_finetune.py --warmup-bias-lr 0.0001 --val-every 10 > data\v2_finetune.log 2>&1
if errorlevel 1 goto fail
echo [%date% %time%] v2: pool tables >> data\status.txt
%PY% make_pool.py --splits val test > data\v2_pool.log 2>&1
if errorlevel 1 goto fail
set CUDA_VISIBLE_DEVICES=

echo [%date% %time%] v2: s05 prepare >> data\status.txt
set PERT_MODE=s05
set PERTINENCE_SMOKE=0
%PY% prepare_pertinence_data.py > data\v2_s05_prepare.log 2>&1
if errorlevel 1 goto fail
echo [%date% %time%] v2: s05 smoke chain >> data\status.txt
set PERTINENCE_SMOKE=1
%PY% 08_pertinence_search.py > data\v2_s05_smoke_search.log 2>&1
if errorlevel 1 goto fail
%PY% 09_pertinence_analysis.py > data\v2_s05_smoke_analysis.log 2>&1
if errorlevel 1 goto fail
%PY% 10_pertinence_report.py > data\v2_s05_smoke_report.log 2>&1
if errorlevel 1 goto fail
echo [%date% %time%] v2: s05 full search >> data\status.txt
set PERTINENCE_SMOKE=0
%PY% 08_pertinence_search.py > data\v2_s05_search.log 2>&1
if errorlevel 1 goto fail
echo [%date% %time%] v2: s05 analysis >> data\status.txt
%PY% 09_pertinence_analysis.py > data\v2_s05_analysis.log 2>&1
if errorlevel 1 goto fail
%PY% 10_pertinence_report.py > data\v2_s05_report.log 2>&1
if errorlevel 1 goto fail
echo [%date% %time%] v2: s05 DONE >> data\status.txt

goto end
:fail
echo [%date% %time%] FAILED - see the last log in data\ >> data\status.txt
:end
