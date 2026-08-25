"""Evidence-closed bound expressions lowered to the exact arithmetic IR."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Mapping, TypeAlias

from .arithmetic_ir import (
    Assign,
    Certificate,
    Constraint,
    Mean,
    Problem,
    Quantity,
    Rate,
    Solution,
    SolveStatus,
    Span,
    Sum,
    Term,
    Unit,
    Variable,
    solve,
)
from .clause_compiler import SymbolKey


EvidenceId: TypeAlias = str


def _exact(value: int | Fraction) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError("expression values must be int or Fraction")
    return Fraction(value)


def _span(value: object) -> None:
    if not isinstance(value, Span):
        raise TypeError("expression spans must be Span instances")


@dataclass(frozen=True, slots=True)
class NumericEvidence:
    evidence_id: EvidenceId
    value: Fraction
    unit: Unit
    span: Span

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_id, str) or not self.evidence_id:
            raise ValueError("evidence_id must be non-empty text")
        object.__setattr__(self, "value", _exact(self.value))
        if not isinstance(self.unit, Unit):
            raise TypeError("numeric evidence unit must be a Unit")
        _span(self.span)


@dataclass(frozen=True, slots=True)
class LiteralExpr:
    value: Fraction
    unit: Unit
    span: Span
    evidence_ids: tuple[EvidenceId, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _exact(self.value))
        if not isinstance(self.unit, Unit):
            raise TypeError("literal unit must be a Unit")
        _span(self.span)
        ids = tuple(self.evidence_ids)
        if any(not isinstance(item, str) or not item for item in ids):
            raise ValueError("literal evidence ids must be non-empty text")
        object.__setattr__(self, "evidence_ids", ids)


@dataclass(frozen=True, slots=True)
class RefExpr:
    symbol: SymbolKey
    span: Span

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, SymbolKey):
            raise TypeError("reference symbol must be a SymbolKey")
        _span(self.span)


@dataclass(frozen=True, slots=True)
class ProductExpr:
    factors: tuple[Expression, ...]
    span: Span

    def __post_init__(self) -> None:
        object.__setattr__(self, "factors", tuple(self.factors))
        _span(self.span)


@dataclass(frozen=True, slots=True)
class SignedTerm:
    sign: int
    expr: Expression
    role: str
    span: Span

    def __post_init__(self) -> None:
        if self.sign not in (-1, 1) or isinstance(self.sign, bool):
            raise ValueError("signed term sign must be exactly -1 or +1")
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("signed term role must be non-empty text")
        _span(self.span)


@dataclass(frozen=True, slots=True)
class SumExpr:
    terms: tuple[SignedTerm, ...]
    span: Span

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        _span(self.span)


@dataclass(frozen=True, slots=True)
class QuotientExpr:
    numerator: Expression
    denominator: LiteralExpr
    span: Span

    def __post_init__(self) -> None:
        if not isinstance(self.denominator, LiteralExpr):
            raise TypeError("quotient denominator must be a ground LiteralExpr")
        _span(self.span)


@dataclass(frozen=True, slots=True)
class MeanExpr:
    values: tuple[Expression, ...]
    span: Span

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        _span(self.span)


Expression: TypeAlias = (
    LiteralExpr | RefExpr | ProductExpr | SumExpr | QuotientExpr | MeanExpr
)


@dataclass(frozen=True, slots=True)
class Definition:
    symbol: SymbolKey
    expr: Expression
    span: Span

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, SymbolKey):
            raise TypeError("definition symbol must be a SymbolKey")
        _span(self.span)


@dataclass(frozen=True, slots=True)
class ExpressionTarget:
    symbol: SymbolKey
    expr: Expression
    span: Span

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, SymbolKey):
            raise TypeError("target symbol must be a SymbolKey")
        _span(self.span)


@dataclass(frozen=True, slots=True)
class ExpressionProgram:
    definitions: tuple[Definition, ...]
    target: ExpressionTarget
    numeric_evidence: tuple[NumericEvidence, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "definitions", tuple(self.definitions))
        object.__setattr__(self, "numeric_evidence", tuple(self.numeric_evidence))
        if not isinstance(self.target, ExpressionTarget):
            raise TypeError("program target must be an ExpressionTarget")


@dataclass(frozen=True, slots=True)
class EvidenceProjection:
    evidence: NumericEvidence
    literal: LiteralExpr
    ast_path: str


@dataclass(frozen=True, slots=True)
class ExpressionCompileResult:
    program: ExpressionProgram
    problem: Problem
    solution: Solution
    evidence_projection: tuple[EvidenceProjection, ...]
    symbol_variables: Mapping[SymbolKey, Variable]

    @property
    def certificate(self) -> Certificate:
        certificate = self.solution.certificate
        if certificate is None:  # impossible after compilation
            raise RuntimeError("compiled expression has no certificate")
        return certificate


class ExpressionCompileError(ValueError):
    """The bound program cannot be represented and proved by the exact IR."""


_EXPRESSION_TYPES = (
    LiteralExpr,
    RefExpr,
    ProductExpr,
    SumExpr,
    QuotientExpr,
    MeanExpr,
)


def _require_expression(expr: object, path: str) -> Expression:
    if not isinstance(expr, _EXPRESSION_TYPES):
        raise ExpressionCompileError(f"{path} is not an expression")
    return expr


def _children(expr: Expression) -> tuple[tuple[str, Expression], ...]:
    if isinstance(expr, (LiteralExpr, RefExpr)):
        return ()
    if isinstance(expr, ProductExpr):
        if len(expr.factors) < 2:
            raise ExpressionCompileError("product requires at least two factors")
        return tuple(
            (f"factor[{index}]", factor) for index, factor in enumerate(expr.factors)
        )
    if isinstance(expr, SumExpr):
        if not expr.terms:
            raise ExpressionCompileError("sum requires at least one signed term")
        result: list[tuple[str, Expression]] = []
        for index, term in enumerate(expr.terms):
            if not isinstance(term, SignedTerm):
                raise ExpressionCompileError(f"sum term[{index}] is not SignedTerm")
            result.append((f"term[{index}]", term.expr))
        return tuple(result)
    if isinstance(expr, QuotientExpr):
        return (("numerator", expr.numerator), ("denominator", expr.denominator))
    if isinstance(expr, MeanExpr):
        if not expr.values:
            raise ExpressionCompileError("mean requires at least one value")
        return tuple(
            (f"value[{index}]", value) for index, value in enumerate(expr.values)
        )
    raise AssertionError("closed expression union exhausted")


def _walk(
    expr: Expression,
    path: str,
) -> tuple[list[tuple[str, LiteralExpr]], set[SymbolKey]]:
    _require_expression(expr, path)
    literals: list[tuple[str, LiteralExpr]] = []
    references: set[SymbolKey] = set()
    if isinstance(expr, LiteralExpr):
        literals.append((path, expr))
    elif isinstance(expr, RefExpr):
        references.add(expr.symbol)
    for label, child in _children(expr):
        child_literals, child_references = _walk(
            _require_expression(child, f"{path}.{label}"), f"{path}.{label}"
        )
        literals.extend(child_literals)
        references.update(child_references)
    return literals, references


def _is_count(unit: Unit) -> bool:
    return unit.dimensions == Unit.count().dimensions


class _Compiler:
    def __init__(self, program: ExpressionProgram) -> None:
        if not isinstance(program, ExpressionProgram):
            raise TypeError("compile_expression expects an ExpressionProgram")
        self.program = program
        self.definitions: dict[SymbolKey, Definition] = {}
        self.direct_references: dict[SymbolKey, set[SymbolKey]] = {}
        self.definition_units: dict[SymbolKey, Unit] = {}
        self.symbol_variables: dict[SymbolKey, Variable] = {}
        self.variables: list[Variable] = []
        self.constraints: list[Constraint] = []
        self.auxiliary_index = 0

    def compile(self) -> ExpressionCompileResult:
        projection = self._validate_structure_and_evidence()
        self._build_variables()
        for definition in self.program.definitions:
            value = self._lower(definition.expr)
            self.constraints.append(
                Assign(self.symbol_variables[definition.symbol], value, definition.span)
            )
        target_value = self._lower(self.program.target.expr)
        target = self.symbol_variables[self.program.target.symbol]
        self.constraints.append(Assign(target, target_value, self.program.target.span))
        problem = Problem(tuple(self.variables), tuple(self.constraints), target)
        solution = solve(problem)
        if solution.status is not SolveStatus.UNIQUE:
            reason = solution.reason or solution.status.value
            raise ExpressionCompileError(
                f"expression has no unique exact solution: {reason}"
            )
        if solution.certificate is None or not solution.certificate.verified:
            raise ExpressionCompileError(
                "expression solution lacks an exact certificate"
            )
        return ExpressionCompileResult(
            program=self.program,
            problem=problem,
            solution=solution,
            evidence_projection=projection,
            symbol_variables=MappingProxyType(dict(self.symbol_variables)),
        )

    def _validate_structure_and_evidence(self) -> tuple[EvidenceProjection, ...]:
        if not self.program.definitions and not isinstance(
            self.program.target, ExpressionTarget
        ):
            raise ExpressionCompileError("program requires exactly one target")
        variable_names: dict[str, SymbolKey] = {}
        all_literals: list[tuple[str, LiteralExpr]] = []
        for index, definition in enumerate(self.program.definitions):
            if not isinstance(definition, Definition):
                raise ExpressionCompileError(f"definition[{index}] is invalid")
            if definition.symbol in self.definitions:
                raise ExpressionCompileError(
                    f"duplicate definition for {definition.symbol.variable_name}"
                )
            self.definitions[definition.symbol] = definition
            prior = variable_names.setdefault(
                definition.symbol.variable_name, definition.symbol
            )
            if prior != definition.symbol:
                raise ExpressionCompileError(
                    "distinct symbols collide after canonicalization"
                )
            literals, references = _walk(
                _require_expression(definition.expr, f"definition[{index}]"),
                f"definition[{index}]",
            )
            all_literals.extend(literals)
            self.direct_references[definition.symbol] = references

        target = self.program.target
        if target.symbol in self.definitions:
            raise ExpressionCompileError("target symbol must not also be a definition")
        prior = variable_names.setdefault(target.symbol.variable_name, target.symbol)
        if prior != target.symbol:
            raise ExpressionCompileError("target collides with a definition variable")
        target_literals, target_references = _walk(
            _require_expression(target.expr, "target"), "target"
        )
        all_literals.extend(target_literals)

        all_references = target_references.union(
            *(references for references in self.direct_references.values())
        )
        undefined = all_references.difference(self.definitions)
        if undefined:
            names = ", ".join(sorted(symbol.variable_name for symbol in undefined))
            raise ExpressionCompileError(f"undefined references: {names}")
        self._validate_dag(target_references)
        return self._validate_evidence(all_literals)

    def _validate_dag(self, target_references: set[SymbolKey]) -> None:
        state: dict[SymbolKey, int] = {}

        def visit(symbol: SymbolKey) -> None:
            marker = state.get(symbol, 0)
            if marker == 1:
                raise ExpressionCompileError(
                    f"cyclic definition at {symbol.variable_name}"
                )
            if marker == 2:
                return
            state[symbol] = 1
            for dependency in self.direct_references[symbol]:
                visit(dependency)
            state[symbol] = 2

        for symbol in self.definitions:
            visit(symbol)

        reachable: set[SymbolKey] = set()

        def connect(symbol: SymbolKey) -> None:
            if symbol in reachable:
                return
            reachable.add(symbol)
            for dependency in self.direct_references[symbol]:
                connect(dependency)

        for symbol in target_references:
            connect(symbol)
        disconnected = set(self.definitions).difference(reachable)
        if disconnected:
            names = ", ".join(sorted(symbol.variable_name for symbol in disconnected))
            raise ExpressionCompileError(
                f"definitions disconnected from target: {names}"
            )

    def _validate_evidence(
        self, literals: list[tuple[str, LiteralExpr]]
    ) -> tuple[EvidenceProjection, ...]:
        evidence: dict[EvidenceId, NumericEvidence] = {}
        for index, item in enumerate(self.program.numeric_evidence):
            if not isinstance(item, NumericEvidence):
                raise ExpressionCompileError(f"numeric_evidence[{index}] is invalid")
            if item.evidence_id in evidence:
                raise ExpressionCompileError(
                    f"duplicate numeric evidence id: {item.evidence_id}"
                )
            evidence[item.evidence_id] = item

        consumed: set[EvidenceId] = set()
        projection: list[EvidenceProjection] = []
        for path, literal in literals:
            if _is_count(literal.unit) and (
                literal.value < 0 or literal.value.denominator != 1
            ):
                raise ExpressionCompileError(
                    f"count literal at {path} must be a non-negative integer"
                )
            if len(literal.evidence_ids) != 1:
                raise ExpressionCompileError(
                    f"literal at {path} must consume exactly one evidence id"
                )
            evidence_id = literal.evidence_ids[0]
            item = evidence.get(evidence_id)
            if item is None:
                raise ExpressionCompileError(
                    f"literal references missing evidence id: {evidence_id}"
                )
            if evidence_id in consumed:
                raise ExpressionCompileError(
                    f"numeric evidence consumed more than once: {evidence_id}"
                )
            if item.value != literal.value:
                raise ExpressionCompileError(
                    f"evidence value does not match literal at {path}"
                )
            if item.unit != literal.unit:
                raise ExpressionCompileError(
                    f"evidence unit does not match literal at {path}"
                )
            if item.span != literal.span:
                raise ExpressionCompileError(
                    f"evidence span does not match literal at {path}"
                )
            consumed.add(evidence_id)
            projection.append(EvidenceProjection(item, literal, path))
        extras = set(evidence).difference(consumed)
        if extras:
            raise ExpressionCompileError(
                "unconsumed numeric evidence: " + ", ".join(sorted(extras))
            )
        return tuple(projection)

    def _unit(self, expr: Expression) -> Unit:
        if isinstance(expr, LiteralExpr):
            return expr.unit
        if isinstance(expr, RefExpr):
            cached = self.definition_units.get(expr.symbol)
            if cached is not None:
                return cached
            definition = self.definitions[expr.symbol]
            unit = self._unit(definition.expr)
            self.definition_units[expr.symbol] = unit
            return unit
        if isinstance(expr, ProductExpr):
            units = [self._unit(factor) for factor in expr.factors]
            result = units[0]
            for unit in units[1:]:
                result = result * unit
            return result
        if isinstance(expr, SumExpr):
            units = [self._unit(term.expr) for term in expr.terms]
            reference = units[0]
            if any(not reference.compatible(unit) for unit in units[1:]):
                raise ExpressionCompileError("sum contains incompatible units")
            return reference
        if isinstance(expr, QuotientExpr):
            if expr.denominator.value == 0:
                raise ExpressionCompileError("quotient denominator must be nonzero")
            return self._unit(expr.numerator) / expr.denominator.unit
        if isinstance(expr, MeanExpr):
            units = [self._unit(value) for value in expr.values]
            reference = units[0]
            if any(not reference.compatible(unit) for unit in units[1:]):
                raise ExpressionCompileError("mean contains incompatible units")
            return reference
        raise AssertionError("closed expression union exhausted")

    def _build_variables(self) -> None:
        for definition in self.program.definitions:
            unit = self._unit(definition.expr)
            variable = Variable(
                definition.symbol.variable_name,
                unit,
                count=_is_count(unit),
                span=definition.span,
            )
            self.symbol_variables[definition.symbol] = variable
            self.variables.append(variable)
        target = self.program.target
        target_unit = self._unit(target.expr)
        variable = Variable(
            target.symbol.variable_name,
            target_unit,
            count=_is_count(target_unit),
            span=target.span,
        )
        self.symbol_variables[target.symbol] = variable
        self.variables.append(variable)

    def _auxiliary(self, unit: Unit, span: Span) -> Variable:
        while True:
            name = f"__signed_expr_{self.auxiliary_index:04d}"
            self.auxiliary_index += 1
            if all(variable.name != name for variable in self.variables):
                break
        variable = Variable(name, unit, count=_is_count(unit), span=span)
        self.variables.append(variable)
        return variable

    def _lower(self, expr: Expression) -> Variable | Quantity:
        if isinstance(expr, LiteralExpr):
            return Quantity(expr.value, expr.unit, expr.span)
        if isinstance(expr, RefExpr):
            return self.symbol_variables[expr.symbol]
        if isinstance(expr, ProductExpr):
            atoms = [self._lower(factor) for factor in expr.factors]
            variables = [atom for atom in atoms if isinstance(atom, Variable)]
            quantities = [atom for atom in atoms if isinstance(atom, Quantity)]
            if len(variables) > 1:
                raise ExpressionCompileError("unknown * unknown is nonlinear")
            if variables:
                accumulator: Variable | Quantity = variables[0]
                remaining = quantities
            else:
                accumulator = quantities[0]
                remaining = quantities[1:]
            for factor in remaining:
                product = self._auxiliary(accumulator.unit * factor.unit, expr.span)
                self.constraints.append(Rate(product, accumulator, factor, expr.span))
                accumulator = product
            return accumulator
        if isinstance(expr, SumExpr):
            atoms = [self._lower(term.expr) for term in expr.terms]
            result = self._auxiliary(self._unit(expr), expr.span)
            terms = tuple(
                Term(atom, Fraction(term.sign))
                for atom, term in zip(atoms, expr.terms, strict=True)
            )
            self.constraints.append(Sum(result, terms, expr.span))
            return result
        if isinstance(expr, QuotientExpr):
            numerator = self._lower(expr.numerator)
            denominator = self._lower(expr.denominator)
            assert isinstance(denominator, Quantity)
            result = self._auxiliary(self._unit(expr), expr.span)
            self.constraints.append(Rate(numerator, result, denominator, expr.span))
            return result
        if isinstance(expr, MeanExpr):
            values = tuple(self._lower(value) for value in expr.values)
            result = self._auxiliary(self._unit(expr), expr.span)
            self.constraints.append(Mean(result, values, expr.span))
            return result
        raise AssertionError("closed expression union exhausted")


def compile_expression(program: ExpressionProgram) -> ExpressionCompileResult:
    """Validate, lower, solve, and exactly certify one bound expression program."""

    return _Compiler(program).compile()
