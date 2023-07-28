import numpy as np
import pytest
from cydrogen import (
    EnergyStore,
    BatteryConnection,
    ElectrolyserCompressorProcess,
    Sun,
    Battery,
    H2Refueler,
    BatteryConnectionToElectrolyserCompressorProcess,
    basic_system,
    HOUR,
    kWH,
    kW,
)


def test_Battery():
    assert Battery(purchased=2000).max_energy_stored == pytest.approx(kWH)
    assert Battery(purchased=3500).max_energy_stored == pytest.approx(4 * kWH)


def test_H2Refueler():
    assert H2Refueler(purchased=1510).max_energy_stored == pytest.approx(kWH)
    assert H2Refueler(purchased=1501.23).max_energy_stored == pytest.approx(0.123 * kWH)


# @pytest.mark.skip("Skipped due to slowness of PVWatts API call.")
def test_Sun():
    s = Sun(purchased=3950)
    assert float(s.data.inputs["system_capacity"]) == pytest.approx(1)
    assert len(s.outputs["ac"]) == 8760
    assert float(
        Sun(purchased=3000 + 7 * 950).data.inputs["system_capacity"]
    ) == pytest.approx(7)


########################################################################################


def test_BatteryConnectionToElectrolyserCompressorProcess():
    proc = BatteryConnectionToElectrolyserCompressorProcess(
        EnergyStore(), EnergyStore(), purchased=3700 + 430
    )
    assert proc.efficiency == pytest.approx(
        ElectrolyserCompressorProcess.efficiency * BatteryConnection.efficiency
    )
    assert proc.max_power_transfer == pytest.approx(kW)
    assert BatteryConnectionToElectrolyserCompressorProcess(
        EnergyStore(), EnergyStore(), purchased=3701
    ).max_power_transfer == pytest.approx(1 / 430 * kW)


# @pytest.mark.skip("Skipped due to slowness.")
def test_max_power_transfer():
    s = basic_system(*[2.5e5] * 4)
    s.simulate()
    d = np.diff(s.us, 1, axis=0)  # need to specify axis=0 for proper behaviour
    mpt = np.array([e.max_power_transfer * HOUR for e in s.g.edges])
    assert np.all(
        (d[:, [s.g.nodes.index(e.node_to) for e in s.g.edges]] <= mpt)
        & (d[:, [s.g.nodes.index(e.node_from) for e in s.g.edges]] >= -mpt)
    )
