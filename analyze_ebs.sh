cd /data/wjlu/work/claude/unfold_fs_demo/fs_test || exit 1
echo "=== test C 的 EBS.dat 结构 ==="
head -6 EBS.dat
echo "..."
awk '!/^#/ {print NF}' EBS.dat | sort | uniq -c
echo "--- K-Path 列取值(前12) ---"
awk '!/^#/ {print $1}' EBS.dat | head -12 | tr '\n' ' '
echo
echo "--- 第1条带的 100 行 (kpath, E, w) 前 10 ---"
awk '!/^#/ {print}' EBS.dat | head -10
echo "--- 权重分布 ---"
awk '!/^#/ && NF==3 {print $3}' EBS.dat | sort -g | uniq -c | head -8
echo "--- 每个 band 块的行数 ---"
awk '/Band-Index/{if(n)print n; n=0; next} {if($0 !~ /^#/) n++} END{print n}' EBS.dat | sort | uniq -c
