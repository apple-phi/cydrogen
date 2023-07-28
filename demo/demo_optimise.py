import numpy as np
import cydrogen
import cydrogen.optimise

# BudgetAllocator demo
opt = cydrogen.optimise.BudgetAllocator(
    {
        "x0": np.ones(3) / 4,
        "bounds": [(0, 1)] * 3,
        "options": {"disp": True},
    }
)
opt.total_budget = 1e6
opt()
"""
>>> # gives the fraction of the first three components of the linear system
>>> # find the last one by subtracting from the total
>>> f(0.6452234719163576, 0.24396066318646292, 0.10691550432184557)=-2094043145527.8591
"""

# MinimiseHEVBudget demo
opt = cydrogen.optimise.MinimiseHEVBudget(
    {
        "x0": np.array([13e3, 14e3, 24e3, 50e3]) * 1000,  # np.ones(4) * 7e5,
        "bounds": [(0, None)] * 4,
        "options": {"disp": True},
    }
)
opt.output_limit = cydrogen.YEARLY_HYUNDAI_NEXO_ENERGY_CONSUMPTION * 1e3
opt()
"""
>>> # Gives minimum budget for each component in a basic linear system to support 1000 HEVs
>>> f(4764064.435943611, 11573739.889519818, 627966.3862981284, 34059.37956929834)=16999830.091330856
"""
