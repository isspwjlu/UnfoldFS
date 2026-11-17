cd /data/wjlu/work/claude/unfold_fs_demo
VK=/data/wjlu/program/vaspkit.1.5.1/bin/vaspkit
mkdir -p fs_test
cd fs_test || exit 1
for f in POSCAR INCAR DOSCAR EIGENVAL; do cp -f ../$f .; done
cp -f ../KPOINTS_mesh10_std KPOINTS
echo "=== vaspkit 262 ==="
$VK -task 262 > vk262.log 2>&1
grep -E "Error|severe|Written|Fermi" vk262.log | head -20
ls -la FERMISURFACE* 2>/dev/null
echo "--- bxsf head ---"
head -30 FERMISURFACE.bxsf 2>/dev/null
cd ..
echo "=== progress ==="
ls -la --time-style=+%H:%M path_run/OUTCAR mesh_run/OUTCAR 2>/dev/null
date
