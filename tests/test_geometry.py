from tests.conftest import FIXTURE_POSCAR


def test_fixture_is_present():
    assert FIXTURE_POSCAR.is_file()
