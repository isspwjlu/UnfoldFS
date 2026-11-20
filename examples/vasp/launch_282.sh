#!/bin/bash
W=/data/wjlu/work/claude/unfold_fs_demo
cd "$W"
nohup bash /tmp/_rsh/run_vaspkit282_all.sh > vk_all.log 2>&1 &
echo "PID=$!"
sleep 240
echo "=== vk_all.log ==="
tail -40 vk_all.log
