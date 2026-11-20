cd /data/wjlu/work/claude/unfold_fs_demo/mesh12
mkdir -p fs_out
nohup /data/wjlu/program/anaconda3/bin/python3 ../prototypes/unfold_procar.py --dir . --outdir fs_out --sigma 0.05 > unfold.log 2>&1 &
echo "PID=$!"
sleep 240
echo "=== unfold.log ==="
cat unfold.log
echo "=== 产物 ==="
ls -la fs_out/
