"""Per-bloc weight bounds for the baseline and DQCP optimizers.

ConstraintSpec captures the full constraint set for a race-bloc coalition
optimization.  It is passed to solve_baseline() and (in Week 5) to the DQCP
optimizer so both share identical simplex + bound logic.

The simplex constraints (sum to 1, non-negativity) are always enforced.
Per-bloc lower and upper bounds tighten the feasible region, e.g.:

    # African American share must be at least 5 %
    spec = ConstraintSpec.from_bounds(
        blocs=CANONICAL_RACES,
        lower={"african_american": 0.05},
    )
"""

from __future__ import annotations

import dataclasses
import math

import cvxpy as cp


@dataclasses.dataclass(frozen=True)
class ConstraintSpec:
    """Simplex constraints plus optional per-bloc lower and upper weight bounds.

    Attributes
    ----------
    blocs:
        Ordered tuple of race-bloc IDs corresponding to the optimizer's
        decision variable vector.  Must match the order used in the covariance
        matrix and mu vector.
    lower:
        Mapping bloc_id → minimum coalition weight.  Missing keys default to
        0.0.  All values must be in [0, 1] and lower[b] <= upper[b].
    upper:
        Mapping bloc_id → maximum coalition weight.  Missing keys default to
        1.0.  All values must be in [0, 1] and lower[b] <= upper[b].
    """

    blocs: tuple[str, ...]
    lower: dict[str, float]
    upper: dict[str, float]

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def default(cls, blocs: list[str] | tuple[str, ...]) -> ConstraintSpec:
        """Unconstrained simplex: w_i in [0, 1] for all i, sum to 1."""
        return cls(blocs=tuple(blocs), lower={}, upper={})

    @classmethod
    def from_bounds(
        cls,
        blocs: list[str] | tuple[str, ...],
        *,
        lower: dict[str, float] | None = None,
        upper: dict[str, float] | None = None,
    ) -> ConstraintSpec:
        """Create a spec with optional per-bloc bounds.

        Any bloc not listed in *lower* defaults to 0.0; any bloc not listed in
        *upper* defaults to 1.0.  Raises ValueError if bounds are infeasible.
        """
        spec = cls(
            blocs=tuple(blocs),
            lower=dict(lower or {}),
            upper=dict(upper or {}),
        )
        spec.validate()
        return spec

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self) -> None:
        """Raise ValueError if the constraint set is internally inconsistent."""
        blocs_set = set(self.blocs)

        for b, lb in self.lower.items():
            if b not in blocs_set:
                raise ValueError(f"ConstraintSpec.lower[{b!r}] is not in blocs {list(self.blocs)}")
            if not math.isfinite(lb) or not (0.0 <= lb <= 1.0):
                raise ValueError(f"ConstraintSpec.lower[{b!r}] = {lb} must be in [0, 1]")

        for b, ub in self.upper.items():
            if b not in blocs_set:
                raise ValueError(f"ConstraintSpec.upper[{b!r}] is not in blocs {list(self.blocs)}")
            if not math.isfinite(ub) or not (0.0 <= ub <= 1.0):
                raise ValueError(f"ConstraintSpec.upper[{b!r}] = {ub} must be in [0, 1]")

        for b in self.blocs:
            lb = self.lower.get(b, 0.0)
            ub = self.upper.get(b, 1.0)
            if lb > ub:
                raise ValueError(f"ConstraintSpec: lower[{b!r}]={lb} > upper[{b!r}]={ub}")

        sum_lb = sum(self.lower.get(b, 0.0) for b in self.blocs)
        if sum_lb > 1.0 + 1e-9:
            raise ValueError(
                f"ConstraintSpec: sum of lower bounds = {sum_lb:.6f} > 1.0; "
                "simplex constraint sum(w)=1 is infeasible."
            )

        sum_ub = sum(self.upper.get(b, 1.0) for b in self.blocs)
        if sum_ub < 1.0 - 1e-9:
            raise ValueError(
                f"ConstraintSpec: sum of upper bounds = {sum_ub:.6f} < 1.0; "
                "simplex constraint sum(w)=1 is infeasible."
            )

    # ── CVXPY constraint builder ──────────────────────────────────────────────

    def cvxpy_constraints(self, w: cp.Variable) -> list:
        """Return the list of CVXPY constraints for weight vector *w*.

        Always includes ``sum(w) == 1``.  Per-bloc lower/upper bounds are added
        only when they tighten the default [0, 1] range, keeping the problem as
        small as possible.

        Parameters
        ----------
        w:
            CVXPY Variable of shape ``(len(self.blocs),)``, declared
            ``nonneg=True`` by the caller so the non-negativity constraint is
            handled efficiently by the solver.
        """
        constraints: list = [cp.sum(w) == 1]
        for i, b in enumerate(self.blocs):
            lb = self.lower.get(b, 0.0)
            ub = self.upper.get(b, 1.0)
            if lb > 0.0:
                constraints.append(w[i] >= lb)
            if ub < 1.0:
                constraints.append(w[i] <= ub)
        return constraints

    # ── Bounds helpers ────────────────────────────────────────────────────────

    def lower_bound(self, bloc: str) -> float:
        """Return the lower bound for *bloc* (0.0 if unspecified)."""
        return self.lower.get(bloc, 0.0)

    def upper_bound(self, bloc: str) -> float:
        """Return the upper bound for *bloc* (1.0 if unspecified)."""
        return self.upper.get(bloc, 1.0)

    def max_achievable_loyalty(self, mu: dict[str, float]) -> float:
        """Greedy upper bound on loyalty achievable within these constraints.

        Assigns as much weight as possible to the highest-loyalty blocs,
        respecting upper bounds, until the simplex fills.  Used by
        solve_baseline() for a fast pre-flight infeasibility check.
        """
        remaining = 1.0 - sum(self.lower_bound(b) for b in self.blocs)
        loyalty = sum(self.lower_bound(b) * mu.get(b, 0.0) for b in self.blocs)

        if remaining <= 1e-12:
            return loyalty

        for b in sorted(self.blocs, key=lambda b: mu.get(b, 0.0), reverse=True):
            slack = self.upper_bound(b) - self.lower_bound(b)
            if slack <= 0.0:
                continue
            alloc = min(remaining, slack)
            loyalty += alloc * mu.get(b, 0.0)
            remaining -= alloc
            if remaining <= 1e-12:
                break
        return loyalty
