"""串联公式批量求值。

一批具名公式可互相引用：先做静态检查（名称冲突、未知引用、依赖环），
再按拓扑依赖序逐条求值。每条公式的结果先换算到它自己的目标单位，
再以该带单位结果放入环境供下游公式引用——绝对温度仍按绝对温度、
温差仍按温差，温度仿射规则与单式求值完全一致。

任何一条公式出错，整批以 422 失败、不返回部分结果；错误 loc 统一
重写到 ``body.formulas.<i>`` 下的 ``expression`` / ``target_unit`` /
``name`` 字段。
"""
from __future__ import annotations

from .errors import EvalError
from .evaluator import Value, convert_result, evaluate, value_from_unit
from .expression import parse_expression, variable_names
from .models import FormulaResult
from .units import ALLOWED_UNITS, parse_target, ureg


def _formula_loc(i: int, field: str) -> tuple[str, ...]:
    return ("body", "formulas", i, field)


def _relocate(exc: EvalError, i: int) -> EvalError:
    """把单式求值的 body.expression / body.target_unit loc 重写到具体公式字段。"""
    loc = list(exc.loc)
    if loc[:2] == ["body", "expression"]:
        loc = [*_formula_loc(i, "expression"), *loc[2:]]
    elif loc[:2] == ["body", "target_unit"]:
        loc = [*_formula_loc(i, "target_unit"), *loc[2:]]
    return EvalError(exc.type, loc, exc.msg)


def _topo_order(refs: list[list[tuple[str, int, int]]]) -> list[int]:
    """迭代 DFS 求拓扑序；灰边（指向求值中的节点）即为依赖环。

    refs[i] 为公式 i 引用的其它公式 (名称, 算式位置, 目标公式下标)。
    """
    n = len(refs)
    state = [0] * n  # 0=未访问 1=求值中(灰) 2=完成(黑)
    order: list[int] = []
    for root in range(n):
        if state[root] != 0:
            continue
        state[root] = 1
        stack: list[list[int]] = [[root, 0]]
        while stack:
            i, edge_i = stack[-1]
            if edge_i < len(refs[i]):
                name, pos, j = refs[i][edge_i]
                stack[-1][1] += 1
                if state[j] == 0:
                    state[j] = 1
                    stack.append([j, 0])
                elif state[j] == 1:
                    raise EvalError(
                        "dependency_cycle",
                        [*_formula_loc(i, "expression"), pos],
                        f"位置 {pos}：公式 {name!r} 与其它公式形成依赖环，"
                        "公式之间的引用必须是无环的有向图",
                    )
            else:
                state[i] = 2
                order.append(i)
                stack.pop()
    return order


def solve_formulas(variables, formulas) -> list[FormulaResult]:
    """校验并按依赖序求值整批公式；任何错误抛带定位的 EvalError。"""
    # --- 名称：公式名须互不重复且不与基础变量重名（标识符规则由 Pydantic 保证）---
    base_names = set(variables)
    formula_index: dict[str, int] = {}
    for i, f in enumerate(formulas):
        if f.name in base_names:
            raise EvalError(
                "formula_name_conflict",
                [*_formula_loc(i, "name")],
                f"公式名 {f.name!r} 与基础变量重名",
            )
        if f.name in formula_index:
            first = formula_index[f.name]
            raise EvalError(
                "formula_name_conflict",
                [*_formula_loc(i, "name")],
                f"公式名 {f.name!r} 重复（此前已在第 {first} 条公式中定义）",
            )
        formula_index[f.name] = i

    # --- 基础变量环境（单位表与单式求值一致）---
    env: dict[str, Value] = {}
    for name, var in variables.items():
        if var.unit not in ALLOWED_UNITS:
            raise EvalError.at_variable(
                "unknown_unit",
                name,
                "unit",
                f"变量 {name!r} 的单位 {var.unit!r} 不受支持，"
                f"可用单位：{', '.join(sorted(ALLOWED_UNITS))}",
            )
        env[name] = value_from_unit(var.value, var.unit)

    # --- 解析全部算式与目标单位（目标单位错误直接定位到该公式字段）---
    asts: list = []
    targets = []
    for i, f in enumerate(formulas):
        try:
            asts.append(parse_expression(f.expression))
        except EvalError as exc:
            raise _relocate(exc, i) from None
        targets.append(parse_target(f.target_unit, _formula_loc(i, "target_unit")))

    # --- 引用：只能引用基础变量或本批公式；同时收集公式间依赖边 ---
    refs: list[list[tuple[str, int, int]]] = []
    for i, f in enumerate(formulas):
        edges = []
        for name, pos in variable_names(asts[i]).items():
            if name in env:
                continue
            if name in formula_index:
                edges.append((name, pos, formula_index[name]))
            else:
                raise EvalError(
                    "unknown_variable",
                    [*_formula_loc(i, "expression"), pos],
                    f"位置 {pos}：公式 {f.name!r} 引用了不存在的变量或公式 "
                    f"{name!r}，可用名称："
                    f"{', '.join(sorted(base_names | set(formula_index)))}",
                )
        refs.append(edges)

    # --- 依赖环检查并取得拓扑求值序（提交顺序不必是计算顺序）---
    order = _topo_order(refs)

    # --- 按依赖序求值：结果换算到各自目标单位后再供下游引用 ---
    results: list[FormulaResult | None] = [None] * len(formulas)
    for i in order:
        f = formulas[i]
        try:
            result = evaluate(asts[i], env)
            magnitude, dimension = convert_result(
                result,
                targets[i],
                loc_prefix=_formula_loc(i, "target_unit"),
                nonfinite_loc_prefix=_formula_loc(i, "expression"),
            )
        except EvalError as exc:
            raise _relocate(exc, i) from None

        # 关键：下游看到的是“按目标单位换算后”的带单位结果
        if result.is_absolute:
            env[f.name] = Value.absolute(magnitude, targets[i].absolute_unit)
        else:
            env[f.name] = Value.general(ureg.Quantity(magnitude, targets[i].pint_unit))

        results[i] = FormulaResult(
            name=f.name,
            value=magnitude,
            unit=targets[i].canonical,
            dimension=dimension,
        )

    return results  # type: ignore[return-value]
