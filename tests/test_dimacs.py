import pytest

from cpq import ganak
from cpq.data import DATA
from cpq.tn import contract
from cpq.vm import read_dimacs


def vm_of(name):
    return read_dimacs(DATA / "dimacs" / f"{name}.dimacs")


def test_berkeleydb_matches_ganak():
    vm = vm_of("berkeleydb")
    assert contract.count(vm, nfeat=vm.nvars) == ganak.count(vm, timeout=300) == 32


@pytest.mark.slow
def test_busybox_matches_ganak():
    vm = vm_of("busybox-1.18.0")
    assert contract.count(vm, nfeat=vm.nvars) == ganak.count(vm, timeout=1800)


@pytest.mark.slow
def test_financial_services_ganak_reference():
    assert ganak.count(vm_of("financial01"), timeout=1800) == 97451212554676
