#!/bin/bash
set -e
echo "=== Fixing Jarvis autostart ==="

[ -f /home/kalch/jarvis.py ] || { echo "Missing jarvis.py"; exit 1; }
[ -f /home/kalch/.env ] || { echo "Missing ~/.env"; exit 1; }

chmod +x /home/kalch/wait_jarvis_hardware.sh /home/kalch/start_jarvis.sh
chmod 600 /home/kalch/.env
chown kalch:kalch /home/kalch/jarvis.py /home/kalch/.env /home/kalch/*.sh 2>/dev/null || true

sudo systemctl stop jarvis 2>/dev/null || true
sudo systemctl unmask jarvis 2>/dev/null || true
sudo cp /home/kalch/jarvis.service /etc/systemd/system/jarvis.service
sudo systemctl daemon-reload
sudo systemctl enable jarvis.service
sudo systemctl start jarvis.service

echo "Done. Logs: tail -f ~/jarvis.log"
