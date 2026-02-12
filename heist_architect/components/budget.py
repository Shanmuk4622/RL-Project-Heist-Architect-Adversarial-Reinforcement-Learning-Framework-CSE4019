"""
Budget System for the Architect agent.
Defines costs for security assets and tracks remaining budget.
"""

from typing import Dict
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Budget Costs
# ---------------------------------------------------------------------------

BUDGET_COSTS: Dict[str, int] = {
    "wall":   1,   # Cheap, static obstacle
    "camera": 3,   # Mid-cost, rotating surveillance
    "guard":  5,   # Expensive, mobile patrol
}

# ---------------------------------------------------------------------------
# Budget Manager
# ---------------------------------------------------------------------------

@dataclass
class BudgetManager:
    """
    Manages the Architect's asset-placement budget.
    
    The Architect has a limited budget to spend on security assets.
    More sophisticated assets (guards, cameras) cost more than walls.
    Budget can scale over training phases for curriculum learning.
    
    Attributes:
        total_budget: Maximum budget available
        spent: Amount already spent
    """
    total_budget: int = 15
    spent: int = 0
    
    @property
    def remaining(self) -> int:
        return self.total_budget - self.spent
    
    def can_afford(self, asset_type: str) -> bool:
        """Check if there's enough budget for an asset type."""
        cost = BUDGET_COSTS.get(asset_type, 0)
        return self.remaining >= cost
    
    def purchase(self, asset_type: str) -> bool:
        """
        Attempt to purchase an asset. Returns True if successful.
        """
        cost = BUDGET_COSTS.get(asset_type, 0)
        if cost == 0:
            return False
        if self.remaining >= cost:
            self.spent += cost
            return True
        return False
    
    def reset(self):
        """Reset the budget for a new level generation."""
        self.spent = 0
    
    def scale_budget(self, new_budget: int):
        """Scale the total budget (for curriculum learning)."""
        self.total_budget = new_budget
        self.spent = 0
    
    def get_affordable_assets(self) -> Dict[str, bool]:
        """Return which assets the Architect can currently afford."""
        return {
            asset: self.can_afford(asset)
            for asset in BUDGET_COSTS
        }
    
    def __repr__(self):
        return (f"Budget(remaining={self.remaining}/{self.total_budget}, "
                f"affordable={self.get_affordable_assets()})")
