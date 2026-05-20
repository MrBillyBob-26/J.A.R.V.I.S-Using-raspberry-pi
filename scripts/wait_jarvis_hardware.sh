#!/bin/bash
MAX_WAIT=90
echo "Waiting for Jarvis USB audio (up to ${MAX_WAIT}s)..."
for ((i = 1; i <= MAX_WAIT; i++)); do
  if grep -qE 'USB PnP|UACDemo|C-Media|PnP Sound|PCM2902' /proc/asound/cards 2>/dev/null; then
    echo "USB audio detected after ${i}s."
    sleep 5
    exit 0
  fi
  sleep 1
done
echo "USB audio not detected; starting anyway."
exit 0
