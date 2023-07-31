import warnings
import traceback
import numpy as np
import abc
import scipy.optimize
from .system import basic_system
from .units import YEARLY_HYUNDAI_NEXO_ENERGY_CONSUMPTION


class Optimiser(abc.ABC):
    BAD_VALUE = 1e20

    def __init__(self, options: dict):
        self.options = options.copy()
        self.options.setdefault("fun", self.objective)
        self.options.setdefault("method", "Nelder-Mead")
        self.options.setdefault("callback", self.callback)

    @abc.abstractmethod
    def objective(self, xs: np.ndarray) -> np.ndarray:
        ...

    def callback(self, intermediate_result):
        print(f"f{tuple(intermediate_result.x)}={intermediate_result.fun}")

    def __call__(self):
        return scipy.optimize.minimize(**self.options)

    optimise = __call__


class BudgetAllocator(Optimiser):
    """Given a total budget, allocate it between the PV, battery, electrolyser and hydrogen storage.

    Example
    -------
    ```py
    opt = BudgetAllocator(
        {
            "x0": np.ones(3) / 4,
            "bounds": [(0, 1)] * 3,
            "options": {"disp": True},
        }
    )
    opt.total_budget = 1e6
    opt()
    ...
    >>> f(0.6452234719163576, 0.24396066318646292, 0.10691550432184557)=-2094043145527.8591
    ```
    """

    total_budget: float | None = None

    def objective(self, xs: np.ndarray):
        pv_spend, battery_spend, electrolyser_spend = xs * self.total_budget
        h2_store_spend = (1 - xs.sum()) * self.total_budget
        if h2_store_spend < 0:
            return self.BAD_VALUE
        try:
            s = basic_system(
                pv_spend, battery_spend, electrolyser_spend, h2_store_spend
            )
            s.simulate()
        except Exception:
            return self.BAD_VALUE

        # maximise output, so minimise -output
        o = s.us[-1, s.g.inspect_ordering().index("OUTPUT")]
        return self.BAD_VALUE if o <= 0 else -o


class MinimiseHEVBudget(Optimiser):
    """Minimise the budget for the hydrogen energy system require to satisfy a given energy demand.

    Example
    -------
    ```py
    opt = MinimiseHEVBudget(
        {
            "x0": np.ones(4) * 7e5,
            "bounds": [(0, None)] * 4,
            "options": {"disp": True},
        }
    )
    opt.total_budget = YEARLY_HYUNDAI_NEXO_ENERGY_CONSUMPTION
    opt()
    ...
    >>> f(13647.045147418976, 14000.255846977234, 24006.483364105225, 49981.911182403564)=101635.695540905
    ```
    """

    output_limit: float

    def objective(self, xs: np.ndarray):
        try:
            s = basic_system(*xs)
            s.simulate()
        except Exception as e:
            warnings.warn(f"Ignoring exception {e} that occurred.")
            traceback.print_exc()
            return self.BAD_VALUE
        return (
            sum(xs)
            if s.us[-1, s.g.inspect_ordering().index("OUTPUT")] > self.output_limit
            else self.BAD_VALUE
        )
