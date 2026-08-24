"""Exact, prompt-independent arithmetic intermediate representation.

The module deliberately starts *after* language understanding.  A caller supplies
typed variables, quantities and constraints; :func:`solve` then proves whether the
target-connected linear system has one exact solution.  It never calls the legacy
``fertig.math`` heuristics and never parses prompt text.

All elimination uses :class:`fractions.Fraction`.  A ``UNIQUE`` result therefore
includes an exact zero-residual certificate rather than a floating-point tolerance.
Expected modelling failures are returned as non-unique statuses, not guessed through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import Iterable, Mapping, TypeAlias


def _fraction(value: int | Fraction) -> Fraction:
    if isinstance(value, bool):
        raise TypeError("boolean values are not arithmetic quantities")
    if not isinstance(value, (int, Fraction)):
        raise TypeError("values must be int or Fraction")
    return Fraction(value)


@dataclass(frozen=True, slots=True)
class Span:
    """Half-open source span used only for provenance, never for inference."""

    start: int
    end: int
    source: str = ""

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("span must satisfy 0 <= start <= end")


Dimensions: TypeAlias = tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class Unit:
    """A rational-scale physical unit with canonical dimension exponents.

    ``scale`` converts a magnitude in this unit to its dimension's base unit.  Thus
    ``Unit("minute", (("time", 1),), 60)`` is compatible with seconds without any
    floating-point conversion.
    """

    symbol: str
    dimensions: Dimensions = ()
    scale: Fraction = Fraction(1)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("unit symbol must not be empty")
        combined: dict[str, int] = {}
        for dimension, exponent in self.dimensions:
            if not dimension or not isinstance(exponent, int):
                raise TypeError(
                    "unit dimensions require string names and integer exponents"
                )
            combined[dimension] = combined.get(dimension, 0) + exponent
        canonical = tuple(
            sorted((name, power) for name, power in combined.items() if power)
        )
        scale = _fraction(self.scale)
        if scale <= 0:
            raise ValueError("unit scale must be positive")
        object.__setattr__(self, "dimensions", canonical)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def scalar(cls) -> Unit:
        return cls("1")

    @classmethod
    def base(cls, dimension: str, *, symbol: str | None = None) -> Unit:
        return cls(symbol or dimension, ((dimension, 1),))

    @classmethod
    def count(cls, *, symbol: str = "count") -> Unit:
        return cls(symbol, (("count", 1),))

    def compatible(self, other: Unit) -> bool:
        return self.dimensions == other.dimensions

    def __mul__(self, other: Unit) -> Unit:
        dimensions = self.dimensions + other.dimensions
        return Unit(
            f"{self.symbol}*{other.symbol}", dimensions, self.scale * other.scale
        )

    def __truediv__(self, other: Unit) -> Unit:
        dimensions = self.dimensions + tuple(
            (name, -power) for name, power in other.dimensions
        )
        return Unit(
            f"{self.symbol}/{other.symbol}", dimensions, self.scale / other.scale
        )


@dataclass(frozen=True, slots=True)
class Variable:
    """An unknown magnitude measured in ``unit``.

    ``count=True`` adds the proof obligation that its unique value is a non-negative
    integer.  It does not silently round a rational solution.
    """

    name: str
    unit: Unit
    count: bool = False
    span: Span | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("variable name must not be empty")
        if self.count and self.unit.dimensions != Unit.count().dimensions:
            raise ValueError("count variables must use a count-compatible unit")


@dataclass(frozen=True, slots=True)
class Quantity:
    """A known exact magnitude."""

    value: Fraction
    unit: Unit
    span: Span | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _fraction(self.value))


Atom: TypeAlias = Variable | Quantity


@dataclass(frozen=True, slots=True)
class Term:
    """A fixed rational coefficient times one atom."""

    atom: Atom
    coefficient: Fraction = Fraction(1)

    def __post_init__(self) -> None:
        if not isinstance(self.atom, (Variable, Quantity)):
            raise TypeError("term atom must be Variable or Quantity")
        object.__setattr__(self, "coefficient", _fraction(self.coefficient))


WeightedAtom: TypeAlias = Atom | Term


@dataclass(frozen=True, slots=True)
class Assign:
    """``target = value``."""

    target: Variable
    value: Atom
    span: Span | None = None


@dataclass(frozen=True, slots=True)
class Affine:
    """``target = scale * source + offset`` with fixed scalar ``scale``."""

    target: Variable
    source: Atom
    scale: Fraction
    offset: Atom
    span: Span | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale", _fraction(self.scale))


@dataclass(frozen=True, slots=True)
class Sum:
    """``total = sum(terms)``; weighted terms support exact ledgers."""

    total: Atom
    terms: tuple[WeightedAtom, ...]
    span: Span | None = None


@dataclass(frozen=True, slots=True)
class Balance:
    """``sum(left) = sum(right)``."""

    left: tuple[WeightedAtom, ...]
    right: tuple[WeightedAtom, ...]
    span: Span | None = None


@dataclass(frozen=True, slots=True)
class Rate:
    """``amount = rate * duration``.

    At least one factor must be a known :class:`Quantity`; multiplying two unknown
    variables is nonlinear and is rejected.
    """

    amount: Atom
    rate: Atom
    duration: Atom
    span: Span | None = None


@dataclass(frozen=True, slots=True)
class Part:
    """``part = fraction * whole`` with a fixed dimensionless fraction."""

    part: Atom
    whole: Atom
    fraction: Fraction
    span: Span | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fraction", _fraction(self.fraction))


@dataclass(frozen=True, slots=True)
class Mean:
    """``mean = sum(values) / len(values)``."""

    mean: Atom
    values: tuple[WeightedAtom, ...]
    span: Span | None = None


Constraint: TypeAlias = Assign | Affine | Sum | Balance | Rate | Part | Mean


@dataclass(frozen=True, slots=True)
class Problem:
    """A closed system and the variable whose connected component is requested."""

    variables: tuple[Variable, ...]
    constraints: tuple[Constraint, ...]
    target: Variable


# Useful descriptive alias for callers that have several IR problem types.
ArithmeticProblem = Problem


class SolveStatus(str, Enum):
    UNIQUE = "unique"
    UNDERDETERMINED = "underdetermined"
    INCONSISTENT = "inconsistent"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class Residual:
    constraint_index: int
    span: Span | None
    value: Fraction


@dataclass(frozen=True, slots=True)
class Certificate:
    """Exact rank and residual evidence for a unique result."""

    rank: int
    variable_count: int
    equation_count: int
    pivot_columns: tuple[int, ...]
    residuals: tuple[Residual, ...]
    component_variables: tuple[str, ...]

    @property
    def verified(self) -> bool:
        return (
            self.rank == self.variable_count
            and len(self.pivot_columns) == self.variable_count
            and all(residual.value == 0 for residual in self.residuals)
        )


@dataclass(frozen=True, slots=True)
class Solution:
    status: SolveStatus
    target_value: Fraction | None = None
    values: Mapping[str, Fraction] = field(default_factory=lambda: MappingProxyType({}))
    certificate: Certificate | None = None
    reason: str = ""

    @property
    def unique(self) -> bool:
        return self.status is SolveStatus.UNIQUE

    def value(self, variable: Variable | str) -> Fraction:
        """Return one proven value, raising if this is not a unique solution."""

        if not self.unique:
            raise ValueError(f"solution is {self.status.value}, not unique")
        name = variable.name if isinstance(variable, Variable) else variable
        return self.values[name]


class _InvalidIR(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _Row:
    coefficients: Mapping[str, Fraction]
    rhs: Fraction
    constraint_index: int
    span: Span | None


def _atom_unit(atom: Atom) -> Unit:
    if not isinstance(atom, (Variable, Quantity)):
        raise _InvalidIR("constraint contains an unsupported atom")
    return atom.unit


def _require_variable(atom: object, role: str) -> Variable:
    if not isinstance(atom, Variable):
        raise _InvalidIR(f"{role} must be a Variable")
    return atom


def _term_parts(term: WeightedAtom) -> tuple[Atom, Fraction]:
    if isinstance(term, Term):
        return term.atom, term.coefficient
    if isinstance(term, (Variable, Quantity)):
        return term, Fraction(1)
    raise _InvalidIR("constraint contains an unsupported weighted term")


def _add_atom(
    coefficients: dict[str, Fraction],
    constant: Fraction,
    atom: Atom,
    multiplier: Fraction,
) -> Fraction:
    """Add an atom expressed in canonical base units."""

    if isinstance(atom, Variable):
        coefficients[atom.name] = coefficients.get(atom.name, Fraction(0)) + (
            multiplier * atom.unit.scale
        )
        if coefficients[atom.name] == 0:
            del coefficients[atom.name]
        return constant
    if isinstance(atom, Quantity):
        return constant + multiplier * atom.value * atom.unit.scale
    raise _InvalidIR("constraint contains an unsupported atom")


def _require_compatible(reference: Unit, atoms: Iterable[WeightedAtom]) -> None:
    for weighted in atoms:
        atom, _ = _term_parts(weighted)
        if not reference.compatible(_atom_unit(atom)):
            raise _InvalidIR(
                f"incompatible units: {reference.symbol} and {_atom_unit(atom).symbol}"
            )


def _linear_equality(
    left: Iterable[tuple[Atom, Fraction]],
    right: Iterable[tuple[Atom, Fraction]],
    *,
    constraint_index: int,
    span: Span | None,
) -> _Row:
    coefficients: dict[str, Fraction] = {}
    constant = Fraction(0)
    for atom, multiplier in left:
        constant = _add_atom(coefficients, constant, atom, multiplier)
    for atom, multiplier in right:
        constant = _add_atom(coefficients, constant, atom, -multiplier)
    return _Row(MappingProxyType(coefficients), -constant, constraint_index, span)


def _constraint_row(constraint: Constraint, index: int) -> _Row:
    span = constraint.span
    if isinstance(constraint, Assign):
        _require_variable(constraint.target, "assignment target")
        _require_compatible(constraint.target.unit, (constraint.value,))
        return _linear_equality(
            ((constraint.target, Fraction(1)),),
            ((constraint.value, Fraction(1)),),
            constraint_index=index,
            span=span,
        )
    if isinstance(constraint, Affine):
        _require_variable(constraint.target, "affine target")
        _require_compatible(
            constraint.target.unit, (constraint.source, constraint.offset)
        )
        return _linear_equality(
            ((constraint.target, Fraction(1)),),
            (
                (constraint.source, constraint.scale),
                (constraint.offset, Fraction(1)),
            ),
            constraint_index=index,
            span=span,
        )
    if isinstance(constraint, Sum):
        _atom_unit(constraint.total)
        if not constraint.terms:
            raise _InvalidIR("sum must contain at least one term")
        _require_compatible(_atom_unit(constraint.total), constraint.terms)
        return _linear_equality(
            ((constraint.total, Fraction(1)),),
            tuple(_term_parts(term) for term in constraint.terms),
            constraint_index=index,
            span=span,
        )
    if isinstance(constraint, Balance):
        if not constraint.left or not constraint.right:
            raise _InvalidIR("balance requires non-empty left and right sides")
        first, _ = _term_parts(constraint.left[0])
        _require_compatible(_atom_unit(first), constraint.left + constraint.right)
        return _linear_equality(
            tuple(_term_parts(term) for term in constraint.left),
            tuple(_term_parts(term) for term in constraint.right),
            constraint_index=index,
            span=span,
        )
    if isinstance(constraint, Rate):
        amount_unit = _atom_unit(constraint.amount)
        rate_unit = _atom_unit(constraint.rate)
        duration_unit = _atom_unit(constraint.duration)
        expected = rate_unit * duration_unit
        if not amount_unit.compatible(expected):
            raise _InvalidIR("amount unit is incompatible with rate * duration")
        if isinstance(constraint.rate, Variable) and isinstance(
            constraint.duration, Variable
        ):
            raise _InvalidIR("unknown * unknown is nonlinear")
        if isinstance(constraint.rate, Variable):
            assert isinstance(constraint.duration, Quantity)
            product = constraint.duration.value * constraint.duration.unit.scale
            right = ((constraint.rate, product),)
        elif isinstance(constraint.duration, Variable):
            assert isinstance(constraint.rate, Quantity)
            product = constraint.rate.value * constraint.rate.unit.scale
            right = ((constraint.duration, product),)
        else:
            assert isinstance(constraint.rate, Quantity)
            assert isinstance(constraint.duration, Quantity)
            product_value = constraint.rate.value * constraint.duration.value
            product_unit = constraint.rate.unit * constraint.duration.unit
            right = ((Quantity(product_value, product_unit), Fraction(1)),)
        return _linear_equality(
            ((constraint.amount, Fraction(1)),),
            right,
            constraint_index=index,
            span=span,
        )
    if isinstance(constraint, Part):
        _atom_unit(constraint.part)
        _require_compatible(constraint.part.unit, (constraint.whole,))
        return _linear_equality(
            ((constraint.part, Fraction(1)),),
            ((constraint.whole, constraint.fraction),),
            constraint_index=index,
            span=span,
        )
    if isinstance(constraint, Mean):
        _atom_unit(constraint.mean)
        if not constraint.values:
            raise _InvalidIR("mean must contain at least one value")
        _require_compatible(_atom_unit(constraint.mean), constraint.values)
        count = Fraction(len(constraint.values))
        return _linear_equality(
            ((constraint.mean, count),),
            tuple(_term_parts(value) for value in constraint.values),
            constraint_index=index,
            span=span,
        )
    raise _InvalidIR(f"unsupported constraint type: {type(constraint).__name__}")


def variables_in_constraint(constraint: Constraint) -> tuple[Variable, ...]:
    """Return every variable referenced by one constraint, including terms."""

    atoms: list[Atom] = []
    if isinstance(constraint, Assign):
        atoms = [constraint.target, constraint.value]
    elif isinstance(constraint, Affine):
        atoms = [constraint.target, constraint.source, constraint.offset]
    elif isinstance(constraint, Sum):
        atoms = [constraint.total, *(_term_parts(term)[0] for term in constraint.terms)]
    elif isinstance(constraint, Balance):
        atoms = [
            *(_term_parts(term)[0] for term in constraint.left),
            *(_term_parts(term)[0] for term in constraint.right),
        ]
    elif isinstance(constraint, Rate):
        atoms = [constraint.amount, constraint.rate, constraint.duration]
    elif isinstance(constraint, Part):
        atoms = [constraint.part, constraint.whole]
    elif isinstance(constraint, Mean):
        atoms = [constraint.mean, *(_term_parts(term)[0] for term in constraint.values)]
    else:
        raise _InvalidIR(f"unsupported constraint type: {type(constraint).__name__}")
    by_name: dict[str, Variable] = {}
    for atom in atoms:
        if isinstance(atom, Variable):
            previous = by_name.setdefault(atom.name, atom)
            if previous != atom:
                raise _InvalidIR(f"conflicting variable definitions for {atom.name}")
    return tuple(by_name.values())


def _variables_in_constraint(constraint: Constraint) -> frozenset[str]:
    return frozenset(variable.name for variable in variables_in_constraint(constraint))


def _target_component(
    target: str, constraint_variables: tuple[frozenset[str], ...]
) -> tuple[frozenset[str], tuple[int, ...]]:
    variables = {target}
    selected: set[int] = set()
    changed = True
    while changed:
        changed = False
        for index, names in enumerate(constraint_variables):
            if index in selected or not names.intersection(variables):
                continue
            selected.add(index)
            before = len(variables)
            variables.update(names)
            changed = changed or len(variables) != before
    return frozenset(variables), tuple(sorted(selected))


def _rref(
    rows: list[list[Fraction]], variable_count: int
) -> tuple[int, tuple[int, ...], bool]:
    pivot_row = 0
    pivots: list[int] = []
    for column in range(variable_count):
        candidate = next(
            (row for row in range(pivot_row, len(rows)) if rows[row][column] != 0),
            None,
        )
        if candidate is None:
            continue
        rows[pivot_row], rows[candidate] = rows[candidate], rows[pivot_row]
        pivot = rows[pivot_row][column]
        rows[pivot_row] = [value / pivot for value in rows[pivot_row]]
        for row in range(len(rows)):
            if row == pivot_row or rows[row][column] == 0:
                continue
            factor = rows[row][column]
            rows[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(rows[row], rows[pivot_row], strict=True)
            ]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(rows):
            break
    inconsistent = any(
        all(value == 0 for value in row[:variable_count]) and row[variable_count] != 0
        for row in rows
    )
    return len(pivots), tuple(pivots), inconsistent


def _failure(status: SolveStatus, reason: str) -> Solution:
    return Solution(status=status, reason=reason)


def solve(problem: Problem) -> Solution:
    """Solve the target-connected component exactly and fail closed.

    ``UNIQUE`` is returned only if all variables in that component have full column
    rank, every original equation has exact zero residual, and all count-domain
    obligations hold.  Disconnected equations cannot influence the requested target.
    """

    try:
        if not isinstance(problem, Problem):
            raise _InvalidIR("solve expects a Problem")
        by_name: dict[str, Variable] = {}
        for variable in problem.variables:
            if not isinstance(variable, Variable):
                raise _InvalidIR("problem variables must be Variable instances")
            if variable.name in by_name:
                raise _InvalidIR(f"duplicate variable name: {variable.name}")
            by_name[variable.name] = variable
        if not isinstance(problem.target, Variable):
            raise _InvalidIR("target must be a Variable")
        declared_target = by_name.get(problem.target.name)
        if declared_target is None or declared_target != problem.target:
            raise _InvalidIR("target must be one of the declared variables")

        constraint_variables = tuple(
            _variables_in_constraint(constraint) for constraint in problem.constraints
        )
        referenced = (
            set().union(*constraint_variables) if constraint_variables else set()
        )
        undeclared = referenced.difference(by_name)
        if undeclared:
            raise _InvalidIR(f"undeclared variables: {', '.join(sorted(undeclared))}")

        # Validate every supplied constraint even though only the target-connected
        # equations participate in rank and consistency.  Disconnected contradictions
        # are irrelevant; malformed IR is not.
        validated_rows = tuple(
            _constraint_row(constraint, index)
            for index, constraint in enumerate(problem.constraints)
        )
        component, selected_indices = _target_component(
            problem.target.name, constraint_variables
        )
        variable_names = tuple(sorted(component))
        original_rows = [validated_rows[index] for index in selected_indices]
        matrix = [
            [row.coefficients.get(name, Fraction(0)) for name in variable_names]
            + [row.rhs]
            for row in original_rows
        ]
        reduced = [row.copy() for row in matrix]
        rank, pivots, inconsistent = _rref(reduced, len(variable_names))
        if inconsistent:
            return _failure(
                SolveStatus.INCONSISTENT,
                "target-connected constraints are contradictory",
            )
        if rank < len(variable_names):
            return _failure(
                SolveStatus.UNDERDETERMINED,
                f"rank {rank} is below {len(variable_names)} component variables",
            )

        values_by_name: dict[str, Fraction] = {}
        for row_index, pivot_column in enumerate(pivots):
            values_by_name[variable_names[pivot_column]] = reduced[row_index][-1]
        if len(values_by_name) != len(variable_names):
            raise _InvalidIR("internal rank certificate is incomplete")

        for name, value in values_by_name.items():
            variable = by_name[name]
            if variable.count and (value.denominator != 1 or value < 0):
                raise _InvalidIR(
                    f"count variable {name} is not a non-negative integer: {value}"
                )

        residuals = tuple(
            Residual(
                constraint_index=row.constraint_index,
                span=row.span,
                value=sum(
                    coefficient * values_by_name[name]
                    for name, coefficient in row.coefficients.items()
                )
                - row.rhs,
            )
            for row in original_rows
        )
        certificate = Certificate(
            rank=rank,
            variable_count=len(variable_names),
            equation_count=len(original_rows),
            pivot_columns=pivots,
            residuals=residuals,
            component_variables=variable_names,
        )
        if not certificate.verified:
            raise _InvalidIR("exact residual certificate failed")
        values = MappingProxyType(dict(sorted(values_by_name.items())))
        return Solution(
            status=SolveStatus.UNIQUE,
            target_value=values[problem.target.name],
            values=values,
            certificate=certificate,
        )
    except (_InvalidIR, TypeError, ValueError) as exc:
        return _failure(SolveStatus.INVALID, str(exc))


__all__ = [
    "Affine",
    "ArithmeticProblem",
    "Assign",
    "Balance",
    "Certificate",
    "Constraint",
    "Mean",
    "Part",
    "Problem",
    "Quantity",
    "Rate",
    "Residual",
    "Solution",
    "SolveStatus",
    "Span",
    "Sum",
    "Term",
    "Unit",
    "Variable",
    "variables_in_constraint",
    "solve",
]
