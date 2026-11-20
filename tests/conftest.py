from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
FIXTURE_POSCAR = DATA_DIR / "POSCAR"
FIXTURE_PRIMCELL = DATA_DIR / "PRIMCELL.vasp"
FIXTURE_TRANSMAT = DATA_DIR / "TRANSMAT.in"
FIXTURE_EIGENVAL = DATA_DIR / "EIGENVAL"
FIXTURE_DOSCAR = DATA_DIR / "DOSCAR"
FIXTURE_PROCAR = DATA_DIR / "PROCAR"


# ---------------------------------------------------------------------------
# 开源版本**不含测试数据**(VASP 的 PROCAR/DOSCAR/EIGENVAL 输出, 体量不小)。
# 缺了就把用到它们的测试跳过, 而不是让整个测试集报错。
# 需要本地跑完整测试时, 把 tests/data/ 补回来即可。
# ---------------------------------------------------------------------------
import pytest  # noqa: E402

_FIXTURE_USERS = {
    "test_expand.py": ("PROCAR",),
    "test_geometry.py": ("DOSCAR", "EIGENVAL", "PROCAR"),
    "test_unfold.py": ("PROCAR",),
    "test_vaspio.py": ("DOSCAR", "PROCAR"),
}


def pytest_collection_modifyitems(config, items):
    missing = {}
    for item in items:
        module = item.nodeid.split("::")[0].split("/")[-1]
        for name in _FIXTURE_USERS.get(module, ()):
            path = DATA_DIR / name
            if not path.exists():
                missing.setdefault(str(path), []).append(item)
    for path, tests in missing.items():
        marker = pytest.mark.skip(reason=f"测试数据未包含在开源版本里: {path}")
        for test in tests:
            test.add_marker(marker)
