set -u
cd /home/agx2/cs24bt033/sched-harness
PY=/home/agx2/cs24bt033/jetson-power/.venv/bin/python
nvpmodel -m 7
nvpmodel -q
sed -i 's/^SESSION_MODE = .*/SESSION_MODE = "MODE_15W_DESKTOP"/' config.py
sed -i 's/^LOCK_CLOCKS = .*/LOCK_CLOCKS = True/' config.py
grep -n "^SESSION_MODE\|^LOCK_CLOCKS" config.py
mkdir -p models logs
cp -n ../jetson-power/models/*.pt models/ 2>/dev/null
$PY 00_preflight.py 2>&1 | tee logs/00_preflight.log
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "PREFLIGHT FAILED"; exit 1; }
if [ -f results/agx_xavier_jp511/profile.json ]; then
  echo "profile.json exists, skipping calibration"
else
  $PY 01_calibrate.py 2>&1 | tee logs/01_calibrate.log
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "CALIBRATION FAILED"; exit 1; }
fi
$PY 02_run.py --smoke 2>&1 | tee logs/02_smoke.log
grep -q "^SMOKE PASS" logs/02_smoke.log || { echo "SMOKE NOT PASSED, plan not started"; exit 2; }
$PY 03_plan.py 2>&1 | tee logs/03_plan.log
echo "ALL DONE"
