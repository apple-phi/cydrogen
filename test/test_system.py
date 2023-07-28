import numpy as np
import pytest
from cydrogen import (
    BatteryConnectionToElectrolyserCompressorProcess,
    ElectrolyserCompressorProcess,
    EnergyStore,
    add_energy_sinks,
)
from cydrogen import EnergySystem

# each case if ordered with (A, U, P, dt)
test_cases = [
    # Test case 1: N = 2
    (
        np.array([[1, 2], [3, 4]]),
        np.array([10, 100]),
        np.array([[0, 5], [12, 1]]),
        7,
    ),
    # Test case 2: N = 3
    (
        np.array([[2, 0, 1], [3, 4, 0], [0, 5, 6]]),
        np.array([1, 2, 3]),
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        np.inf,
    ),
    # Test case 3: N = 2
    (
        np.array([[1, 1], [1, 1]]),
        np.array([5, 5]),
        np.array([[0, 0], [0, 0]]),
        0,
    ),
    # Test case 4: N = 4
    (
        np.array(
            [[np.inf, 1, 2, 3], [4, 5, 6, 7], [8, 9, np.inf, 10], [11, 12, 13, np.inf]]
        ),
        np.array([10, 20, 30, 40]),
        np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]),
        5,
    ),
    # Test case 5: N = 3
    (
        np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]]),
        np.array([1, 2, 3]),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        10,
    ),
    # Test case 6: N = 5
    (
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        ),
        np.array([10, 20, 30, 40, 50]),
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ),
        2,
    ),
    # Test case 7: N = 4
    (
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        np.array([1, 1, 1, 1]),
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        1,
    ),
    # Test case 8: N = 2
    (
        np.array([[0, 1], [1, 0]]),
        np.array([1, 2]),
        np.array([[np.inf, 1], [1, np.inf]]),
        3,
    ),
    # Test case 9: N = 3
    (
        np.array([[0, 2, 0], [1, 0, 3], [0, 4, 0]]),
        np.array([0, 0, 1]),
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        6,
    ),
    # Test case 10: N = 5
    (
        np.array(
            [
                [5, 0, 0, 0, 0],
                [0, 4, 0, 0, 0],
                [0, 0, 3, 0, 0],
                [0, 0, 0, 2, 0],
                [0, 0, 0, 0, 1],
            ]
        ),
        np.array([1, 2, 3, 4, 5]),
        np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        ),
        5,
    ),
    # (
    #     np.array(
    #         [
    #             [0.0, 1.0, 0.0, 1.0],
    #             [0.0, 0.0, 1.0, 0.0],
    #             [0.0, 0.0, 0.0, 0.0],
    #             [0.0, 1.0, 0.0, 0.0],
    #         ]
    #     ),
    #     np.array([np.inf, 0.0, -np.inf, 0.0]),
    #     np.array(
    #         [
    #             [0.0, 100.0, 0.0, 100.0],
    #             [0.0, 0.0, np.inf, 0.0],
    #             [0.0, 0.0, 0.0, 0.0],
    #             [0.0, np.inf, 0.0, 0.0],
    #         ]
    #     ),
    #     3600,
    # ),
]


@pytest.mark.parametrize("A, U, P, dt", test_cases)
def test_dU(A, U, P, dt):
    du_vectorized = EnergySystem._dU(A, U, P, dt)
    du_naive = np.array(
        [
            -sum(min(A[i, j] * U[i], dt * P[i, j]) for j in range(U.size))
            + sum(min(A[j, i] * U[j], dt * P[j, i]) for j in range(U.size))
            for i in range(U.size)
        ]
    )
    assert np.allclose(du_vectorized, du_naive)


def test_tiny_system():
    in_ = EnergyStore(value=1, name="INPUT")
    out = EnergyStore(value=0, name="OUTPUT")
    in_.link_to(out, weight=1, using=ElectrolyserCompressorProcess, purchased=1e20)
    s = EnergySystem(add_energy_sinks(in_.to_graph()))
    # s.g.nx_show()
    s.simulate()
    assert s.us[-1].max() == pytest.approx(0.7, abs=0.01)  # output
    assert s.us[-1][0] == pytest.approx(0)  # input
    assert sorted(s.us[-1])[1] == pytest.approx(0.3, abs=0.01)  # sink


def test_purchasing():
    in_ = EnergyStore(value=1, name="INPUT")
    out = EnergyStore(value=0, name="OUTPUT")
    in_.link_to(
        out,
        weight=1,
        using=BatteryConnectionToElectrolyserCompressorProcess,
        purchased=3701,
    )
    e = list(in_.edges_out)[0]
    assert e.purchased == 3701
    assert 1 < e.max_power_transfer < 1e6
