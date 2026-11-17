"""单元测试: PROCAR / EIGENVAL / POSCAR 解析器"""
import sys
sys.path.insert(0, '/data/wjlu/work/claude/unfold_fs_demo')
import numpy as np
import unfold_procar as U

print("=== EIGENVAL 解析 ===")
k, e = U.read_eigenval('/data/wjlu/work/claude/unfold_fs_demo/EIGENVAL')
print(" k 点:", k.shape, " 能量:", e.shape)
print(" 前3个 k:", k[:3].tolist())
print(" band1 能量前5:", e[:5, 0].tolist())

print("=== POSCAR 解析 ===")
lat, sp, pos = U.read_poscar('/data/wjlu/work/claude/unfold_fs_demo/POSCAR')
print(" lattice:", lat.tolist())
print(" species:", sp[:6], "...")
print(" 第1个原子(分数):", pos[0].tolist())

print("=== PROCAR 解析 (Bi2Se3, LORBIT=14, 应无相位列) ===")
import itertools
src = open('/data/wjlu/work/claude/unfold_fs_demo/unfold_procar.py').read()
# 只读前 2 个 k 点以节省时间: 临时改写 read_procar 的 nk
lines = open('/data/wjlu/work/vasp_work/Bi2Se3/PROCAR').readlines()[:4000]
open('/tmp/procar_head.txt', 'w').writelines(lines)
projs, wts = U.read_procar('/tmp/procar_head.txt')
print(" projs:", projs.shape, " wts:", wts.shape)
print(" 第1带第1离子复投影:", projs[0, 0, 0])
print(" 权重 tot:", wts[0, :3])
