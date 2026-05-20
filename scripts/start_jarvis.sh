#!/bin/bash
LOG="$HOME/jarvis.log"
exec >>"$LOG" 2>&1
echo "========== $(date) Jarvis boot =========="

export HOME=/home/kalch
cd /home/kalch || exit 1

if [ -f /home/kalch/.env ]; then
  set -a
  . /home/kalch/.env
  set +a
  echo ".env loaded."
else
  echo "ERROR: /home/kalch/.env missing"
  exit 1
fi

/home/kalch/wait_jarvis_hardware.sh

echo "Waiting for network..."
for ((i = 1; i <= 90; i++)); do
  if ping -c 1 -W 2 1.1.1.1 >/dev/null 2>&1; then
    echo "Network up after ${i}s."
    sleep 3
    break
  fi
  sleep 1
done

source /home/kalch/jarvis_env/bin/activate
exec /home/kalch/jarvis_env/bin/python /home/kalch/jarvis.py
