"""POST /v1/formulas 的行为测试：具名公式串联批量求值。"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def formulas(body):
    return client.post("/v1/formulas", json=body)


def ok(variables, formula_specs):
    """formula_specs: [(name, expression, target_unit), ...]（即提交顺序）。"""
    resp = formulas(
        {
            "variables": variables,
            "formulas": [
                {"name": n, "expression": e, "target_unit": u}
                for n, e, u in formula_specs
            ],
        }
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def err(variables, formula_specs):
    resp = formulas(
        {
            "variables": variables,
            "formulas": [
                {"name": n, "expression": e, "target_unit": u}
                for n, e, u in formula_specs
            ],
        }
    )
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, list) and detail, resp.text
    return detail[0]


def by_name(results):
    return {r["name"]: r for r in results}


# ---------------------------------------------------------------------------
# 成功：按提交顺序返回；提交顺序不必是计算顺序
# ---------------------------------------------------------------------------

def test_results_returned_in_submission_order():
    results = ok(
        {"d": {"value": 36, "unit": "km"}, "t": {"value": 2, "unit": "h"},
         "dur": {"value": 1, "unit": "min"}},
        [
            ("distance", "speed * dur", "m"),  # 依赖 speed，却先提交
            ("speed", "d / t", "m/s"),
        ],
    )
    assert [r["name"] for r in results] == ["distance", "speed"]
    got = by_name(results)
    assert got["speed"]["value"] == pytest.approx(5.0)
    assert got["speed"]["unit"] == "m/s"
    assert got["speed"]["dimension"] == "[length] / [time]"
    assert got["distance"]["value"] == pytest.approx(300.0)
    assert got["distance"]["unit"] == "m"


def test_diamond_dependency_chain():
    # total = dist + extra；speed = dist / time；distance 依赖 speed
    results = ok(
        {
            "dist": {"value": 36, "unit": "km"},
            "time": {"value": 2, "unit": "h"},
            "extra": {"value": 10, "unit": "km"},
            "dur": {"value": 1, "unit": "h"},
        },
        [
            ("total", "dist + extra", "m"),
            ("speed", "dist / time", "m/s"),
            ("distance", "speed * dur", "m"),
        ],
    )
    got = by_name(results)
    assert got["total"]["value"] == pytest.approx(46000.0)
    assert got["speed"]["value"] == pytest.approx(5.0)
    assert got["distance"]["value"] == pytest.approx(18000.0)


def test_downstream_uses_upstream_converted_to_its_target_unit():
    # speed 以 m/s 为目标；1 小时的距离应为 5 m/s * 3600 s = 18000 m，
    # 而不是 36 km（说明用的是按目标单位换算后的 speed，而非原始数值 36/2）。
    results = ok(
        {"d": {"value": 36, "unit": "km"}, "t": {"value": 2, "unit": "h"},
         "dur": {"value": 1, "unit": "h"}},
        [
            ("speed", "d / t", "m/s"),
            ("distance", "speed * dur", "m"),
        ],
    )
    assert results[1]["value"] == pytest.approx(18000.0)
    assert results[1]["unit"] == "m"
    assert results[1]["dimension"] == "[length]"


def test_chain_is_self_consistent_across_upstream_target_choices():
    # 同一算式 speed=d/t，上游无论选 m/s 还是 km/h，下游物理结果都自洽相同：
    # 5 m/s * 1 h 与 18 km/h * 1 h 都是 18 km（换算均沿目标单位进行）。
    def distance_in_meters(target):
        results = ok(
            {"d": {"value": 36, "unit": "km"}, "t": {"value": 2, "unit": "h"},
             "dur": {"value": 1, "unit": "h"}},
            [
                ("speed", "d / t", target),
                ("distance", "speed * dur", "m"),
            ],
        )
        return results[1]["value"]

    assert distance_in_meters("m/s") == pytest.approx(18000.0)
    assert distance_in_meters("km/h") == pytest.approx(18000.0)


def test_temperature_chain_absolute_and_delta_stay_distinct():
    # z = t0 换算到 degF（50）；dt = t0 - z 是两个绝对温度相减 => 温差 0 delta_K；
    # t1 = t0 + dt 仍是绝对温度。绝对温度与温差的区分沿链保持。
    results = ok(
        {"t0": {"value": 10, "unit": "degC"}},
        [
            ("dt", "t0 - z", "delta_K"),
            ("z", "t0", "degF"),
            ("t1", "t0 + dt", "degF"),
        ],
    )
    got = by_name(results)
    assert got["dt"]["value"] == pytest.approx(0.0)
    assert got["dt"]["unit"] == "delta_K"
    assert got["dt"]["dimension"] == "[temperature]"
    assert got["z"]["value"] == pytest.approx(50.0)
    assert got["z"]["unit"] == "degF"
    assert got["t1"]["value"] == pytest.approx(50.0)
    assert got["t1"]["unit"] == "degF"


def test_delta_temperature_chain_and_conversion():
    # rise（绝对-绝对 => 温差）换算成 delta_degF 后，再加到绝对温度上。
    results = ok(
        {"t0": {"value": 0, "unit": "degC"}, "t1": {"value": 10, "unit": "degC"}},
        [
            ("rise", "t1 - t0", "delta_degF"),
            ("t2", "t0 + rise", "degF"),
        ],
    )
    got = by_name(results)
    assert got["rise"]["value"] == pytest.approx(18.0)
    assert got["rise"]["unit"] == "delta_degF"
    assert got["t2"]["value"] == pytest.approx(32.0 + 18.0)


def test_dimensionless_formula_chain():
    results = ok(
        {},
        [
            ("a", "2 * 3", "1"),
            ("b", "a + 4", "1"),
        ],
    )
    assert by_name(results)["b"]["value"] == 10.0
    assert results[1]["unit"] == "1"
    assert results[1]["dimension"] == "dimensionless"


def test_single_formula_is_allowed():
    results = ok({"x": {"value": 1, "unit": "m"}}, [("a", "x", "cm")])
    assert results[0]["value"] == pytest.approx(100.0)
    assert results[0]["name"] == "a"


def test_formula_can_reference_formula_result_more_than_once():
    results = ok(
        {},
        [
            ("x", "2 + 3", "1"),
            ("y", "x*x - 3*x + 2", "1"),
        ],
    )
    assert by_name(results)["y"]["value"] == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# 422：依赖环
# ---------------------------------------------------------------------------

def test_two_formula_cycle_locates_expression():
    e = err(
        {},
        [
            ("a", "b", "1"),
            ("b", "a", "1"),
        ],
    )
    assert e["type"] == "dependency_cycle"
    # 环在第 2 条公式（下标 1）对 a（算式位置 0）的引用处闭合
    assert e["loc"] == ["body", "formulas", 1, "expression", 0]


def test_self_dependency_is_a_cycle():
    e = err({}, [("a", "a + 1", "1")])
    assert e["type"] == "dependency_cycle"
    assert e["loc"] == ["body", "formulas", 0, "expression", 0]


def test_three_formula_cycle():
    # a -> c -> b -> a：DFS 顺序 a→c→b，回边在 b 对 a 的引用处闭合
    e = err(
        {},
        [
            ("a", "c + 1", "1"),
            ("b", "a", "1"),
            ("c", "b", "1"),
        ],
    )
    assert e["type"] == "dependency_cycle"
    assert e["loc"] == ["body", "formulas", 1, "expression", 0]


# ---------------------------------------------------------------------------
# 422：引用不存在
# ---------------------------------------------------------------------------

def test_reference_to_unknown_name_locates_position():
    e = err({}, [("a", "x + 1", "1")])
    assert e["type"] == "unknown_variable"
    assert e["loc"] == ["body", "formulas", 0, "expression", 0]


def test_unknown_reference_in_second_formula():
    e = err(
        {"x": {"value": 1, "unit": "m"}},
        [
            ("a", "x", "m"),
            ("b", "a + nope", "m"),
        ],
    )
    assert e["type"] == "unknown_variable"
    assert e["loc"] == ["body", "formulas", 1, "expression", 4]


def test_formula_name_alone_is_a_valid_reference_source():
    # 即使下游目标单位不兼容，引用本身成立；这里仅验证名称解析
    e = err(
        {},
        [
            ("a", "1", "1"),
            ("b", "a + 2", "m"),
        ],
    )
    assert e["type"] == "incompatible_units"
    assert e["loc"] == ["body", "formulas", 1, "target_unit"]


# ---------------------------------------------------------------------------
# 422：名称冲突与请求形状
# ---------------------------------------------------------------------------

def test_formula_name_conflicts_with_base_variable():
    e = err(
        {"d": {"value": 1, "unit": "m"}},
        [("d", "1", "1")],
    )
    assert e["type"] == "formula_name_conflict"
    assert e["loc"] == ["body", "formulas", 0, "name"]


def test_duplicate_formula_names_locates_later_one():
    resp = formulas(
        {
            "variables": {},
            "formulas": [
                {"name": "a", "expression": "1", "target_unit": "1"},
                {"name": "a", "expression": "2", "target_unit": "1"},
            ],
        }
    )
    assert resp.status_code == 422
    e = resp.json()["detail"][0]
    assert e["type"] == "formula_name_conflict"
    assert e["loc"] == ["body", "formulas", 1, "name"]


def test_illegal_formula_name_shape_is_pydantic_422():
    resp = formulas(
        {"variables": {}, "formulas": [{"name": "1a", "expression": "1", "target_unit": "1"}]}
    )
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"][:4] == ["body", "formulas", 0, "name"]


def test_at_least_one_formula_is_required():
    resp = formulas({"variables": {"x": {"value": 1, "unit": "m"}}, "formulas": []})
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "formulas"]


def test_missing_fields_are_422():
    assert formulas({}).status_code == 422
    assert formulas({"variables": {}}).status_code == 422


def test_non_finite_base_value_is_422():
    resp = formulas(
        {
            "variables": {"x": {"value": "nan", "unit": "m"}},
            "formulas": [{"name": "a", "expression": "x", "target_unit": "m"}],
        }
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 422：解析 / 求值 / 换算错误都定位到具体公式
# ---------------------------------------------------------------------------

def test_syntax_error_locates_formula_expression():
    e = err(
        {},
        [
            ("a", "1 + 2", "1"),
            ("b", "1 & 2", "1"),
        ],
    )
    assert e["type"] == "syntax_error"
    assert e["loc"] == ["body", "formulas", 1, "expression", 2]


def test_division_by_zero_in_upstream_locates_that_formula():
    e = err(
        {"x": {"value": 0, "unit": "s"}},
        [
            ("a", "1 / x", "1/s"),
            ("b", "a * 2", "1/s"),
        ],
    )
    assert e["type"] == "division_by_zero"
    assert e["loc"] == ["body", "formulas", 0, "expression", 2]


def test_division_by_zero_through_formula_reference():
    e = err(
        {},
        [
            ("z", "1 - 1", "1"),
            ("r", "1 / z", "1"),
        ],
    )
    assert e["type"] == "division_by_zero"
    assert e["loc"] == ["body", "formulas", 1, "expression", 2]


def test_incompatible_addition_locates_formula_operator():
    e = err(
        {"d": {"value": 1, "unit": "m"}, "t": {"value": 1, "unit": "s"}},
        [("a", "d + t", "m")],
    )
    assert e["type"] == "incompatible_units"
    assert e["loc"] == ["body", "formulas", 0, "expression", 2]


def test_downstream_dimension_mismatch_locates_downstream_formula():
    e = err(
        {"d": {"value": 36, "unit": "km"}, "t": {"value": 2, "unit": "h"}},
        [
            ("speed", "d / t", "m/s"),
            ("bad", "speed + d", "m"),
        ],
    )
    assert e["type"] == "incompatible_units"
    assert e["loc"] == ["body", "formulas", 1, "expression", 6]


def test_target_unit_conversion_error_locates_formula_target_field():
    e = err(
        {"d": {"value": 1, "unit": "m"}},
        [
            ("a", "d", "m"),
            ("b", "d", "s"),
        ],
    )
    assert e["type"] == "incompatible_units"
    assert e["loc"] == ["body", "formulas", 1, "target_unit"]


def test_invalid_target_unit_syntax_locates_position_in_formula():
    e = err(
        {"d": {"value": 1, "unit": "m"}},
        [("a", "d", "m/")],
    )
    assert e["type"] == "invalid_target_unit"
    assert e["loc"][:4] == ["body", "formulas", 0, "target_unit"]


def test_unknown_target_unit_symbol_locates_formula_target():
    e = err(
        {"d": {"value": 1, "unit": "m"}},
        [("a", "d", "mile")],
    )
    assert e["type"] == "unknown_unit"
    assert e["loc"][:4] == ["body", "formulas", 0, "target_unit"]


def test_absolute_temperature_muldiv_rule_holds_across_chain():
    # 上游产出绝对温度，下游若对其取乘除仍按原规则报 422
    e = err(
        {"t": {"value": 10, "unit": "degC"}},
        [
            ("ta", "t", "K"),
            ("bad", "ta * 2", "K"),
        ],
    )
    assert e["type"] == "absolute_temperature"
    assert e["loc"][:4] == ["body", "formulas", 1, "expression"]


def test_absolute_result_requires_absolute_target():
    e = err(
        {"t": {"value": 10, "unit": "degC"}},
        [("ta", "t", "delta_K")],
    )
    assert e["type"] == "incompatible_units"
    assert e["loc"] == ["body", "formulas", 0, "target_unit"]


def test_unknown_base_variable_unit_locates_variable_field():
    e = err(
        {"x": {"value": 1, "unit": "mile"}},
        [("a", "x", "m")],
    )
    assert e["type"] == "unknown_unit"
    assert e["loc"] == ["body", "variables", "x", "unit"]


def test_non_finite_result_locates_formula_expression():
    e = err(
        {"x": {"value": 1e308, "unit": "m"}},
        [("a", "x * x", "m^2")],
    )
    assert e["type"] == "non_finite_result"
    assert e["loc"] == ["body", "formulas", 0, "expression"]


def test_batch_failure_returns_no_partial_results():
    resp = formulas(
        {
            "variables": {},
            "formulas": [
                {"name": "a", "expression": "1", "target_unit": "1"},
                {"name": "b", "expression": "1 / (a - a)", "target_unit": "1"},
            ],
        }
    )
    assert resp.status_code == 422
    body = resp.json()
    assert "detail" in body and isinstance(body["detail"], list)
    assert body["detail"][0]["type"] == "division_by_zero"


# ---------------------------------------------------------------------------
# 既有接口保持不变
# ---------------------------------------------------------------------------

def test_existing_endpoints_still_work():
    r = client.post(
        "/v1/evaluate",
        json={
            "expression": "dist / time",
            "variables": {"dist": {"value": 36, "unit": "km"},
                          "time": {"value": 2, "unit": "h"}},
            "target_unit": "m/s",
        },
    )
    assert r.status_code == 200
    assert r.json()["value"] == pytest.approx(5.0)

    assert client.get("/healthz").status_code == 200
