P=$WORKDIR/claude/unfold_fs_demo/lo12/PROCAR
echo "=== 行 30-50 (相位块开始?) ==="
sed -n '30,50p' "$P" | cat -n | cut -c1-180
echo
echo "=== 20 字段的行样例 ==="
awk 'NF==20 {print NR": "$0; c++; if(c>=3) exit}' "$P" | cut -c1-200
echo
echo "=== 10 字段的行样例 ==="
awk 'NF==10 {print NR": "$0; c++; if(c>=3) exit}' "$P" | cut -c1-200
echo
echo "=== 结构: 每行字段数随行号变化 ==="
awk 'NR<80 {print NR": "NF}' "$P" | tr '\n' ' '
