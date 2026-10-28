@echo off
rem Generated launcher. Logs go to data\*.log, progress lines to data\status.txt.
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set CUDA_VISIBLE_DEVICES=
set CF_TAG=
echo [%date% %time%] v1 start >> data\status.txt
%PY% -m pip install pymoo > data\pip_pymoo.log 2>&1
%PY% -c "import pymoo; print('pymoo', pymoo.__version__)" >> data\pip_pymoo.log 2>&1
if errorlevel 1 goto fail

echo [%date% %time%] v1: s05 prepare >> data\status.txt
set PERT_MODE=s05
set PERTINENCE_SMOKE=0
%PY% prepare_pertinence_data.py > data\v1_s05_prepare.log 2>&1
if errorlevel 1 goto fail
echo [%date% %time%] v1: s05 smoke chain >> data\status.txt
set PERTINENCE_SMOKE=1
%PY% 08_pertinence_search.py > data\v1_s05_smoke_search.log 2>&1
if errorlevel 1 goto fail
%PY% 09_pertinence_analysis.py > data\v1_s05_smoke_analysis.log 2>&1
if errorlevel 1 goto fail
%PY% 10_pertinence_report.py > data\v1_s05_smoke_report.log 2>&1
if errorlevel 1 goto fail
echo [%date% %time%] v1: s05 full search >> data\status.txt
set PERTINENCE_SMOKE=0
%PY% 08_pertinence_search.py > data\v1_s05_search.log 2>&1
if errorlevel 1 goto fail
echo [%date% %time%] v1: s05 analysis >> data\status.txt
%PY% 09_pertinence_analysis.py > data\v1_s05_analysis.log 2>&1
if errorlevel 1 goto fail
%PY% 10_pertinence_report.py > data\v1_s05_report.log 2>&1
if errorlevel 1 goto fail
echo [%date% %time%] v1: s05 DONE >> data\status.txt

echo [%date% %time%] v1: train prepare >> data\status.txt
set PERT_MODE=train
set PERTINENCE_SMOKE=0
%PY% prepare_pertinence_data.py > data\v1_train_prepare.log 2>&1
if errorlevel 1 goto fail
echo [%date% %time%] v1: train smoke chain >> data\status.txt
set PERTINENCE_SMOKE=1
%PY% 08_pertinence_search.py > data\v1_train_smoke_search.log 2>&1
if errorlevel 1 goto fail
%PY% 09_pertinence_analysis.py > data\v1_train_smoke_analysis.log 2>&1
if errorlevel 1 goto fail
%PY% 10_pertinence_report.py > data\v1_train_smoke_report.log 2>&1
if errorlevel 1 goto fail
echo [%date% %time%] v1: train full search >> data\status.txt
set PERTINENCE_SMOKE=0
%PY% 08_pertinence_search.py > data\v1_train_search.log 2>&1
if errorlevel 1 goto fail
echo [%date% %time%] v1: train analysis >> data\status.txt
%PY% 09_pertinence_analysis.py > data\v1_train_analysis.log 2>&1
if errorlevel 1 goto fail
%PY% 10_pertinence_report.py > data\v1_train_report.log 2>&1
if errorlevel 1 goto fail
echo [%date% %time%] v1: train DONE >> data\status.txt

goto end
:fail
echo [%date% %time%] FAILED - see the last log in data\ >> data\status.txt
:end
