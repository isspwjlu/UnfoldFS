cd /data/wjlu/work/claude/unfold_fs_demo/fs_test || exit 1
VK=/data/wjlu/program/vaspkit.1.5.1/bin/vaspkit
cp -f ../KPOINTS_mesh10 KPOINTS
cp -f ../KPOINTS_MAPPING_TABLE_mesh10.in KPOINTS_MAPPING_TABLE.in
ln -sf ../mesh_run2/WAVECAR WAVECAR
ln -sf ../mesh_run2/DOSCAR DOSCAR
ln -sf ../mesh_run2/EIGENVAL EIGENVAL
ln -sf ../mesh_run2/TRANSMAT.in TRANSMAT.in
ln -sf ../mesh_run2/PRIMCELL.vasp PRIMCELL.vasp
ls -la WAVECAR

echo "############ A) 无 KPATH.in ############"
rm -f KPATH.in EBS.dat
$VK -task 282 > vk_noKpath.log 2>&1
grep -E "Error|severe|NKPTS|Percentage" vk_noKpath.log | head -6
head -4 EBS.dat 2>/dev/null

echo
echo "############ B) 官方 KPATH.in (2 段) ############"
cp -f ../KPATH.in .
rm -f EBS.dat
$VK -task 282 > vk_kpath2.log 2>&1
grep -E "Error|severe|NKPTS|Percentage" vk_kpath2.log | head -6
head -4 EBS.dat 2>/dev/null

echo
echo "############ C) 100 段 KPATH.in ############"
{
  echo "Monkhorst-Pack"
  echo "20"
  echo "Line mode"
  echo "Rec"
  for i in $(seq 0 99); do
    x=$(awk -v i=$i 'BEGIN{printf "%.6f", i/100.0}')
    echo "  $x 0.00000000000 0.00000000000 # P$i"
    echo ""
    echo "  $x 0.01000000000 0.00000000000 # P$i"
  done
} > KPATH.in
rm -f EBS.dat
$VK -task 282 > vk_kpath100.log 2>&1
grep -E "Error|severe|NKPTS|Percentage" vk_kpath100.log | head -6
head -4 EBS.dat 2>/dev/null
grep -m1 'NKPTS' EBS.dat 2>/dev/null
