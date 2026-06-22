from model_contamination.benchmarks import load_registry
from model_contamination.types import Verdict


def test_registry_loads():
    reg = load_registry()
    assert len(reg.all()) > 0


def test_gsm8k_present():
    reg = load_registry()
    spec = reg.get("gsm8k")
    assert spec.family == "gsm"
    assert spec.format == "math_cot"
    assert spec.trustworthiness_default == Verdict.SUSPECT


def test_variant_references_valid():
    """所有 variants 字段引用的 benchmark 必须也在注册表中。"""
    reg = load_registry()
    for spec in reg.all():
        for v in spec.variants:
            reg.get(v)  # 抛 KeyError 即测试失败


def test_get_family():
    reg = load_registry()
    gsm_family = reg.get_family("gsm")
    names = {s.name for s in gsm_family}
    assert "gsm8k" in names
    assert "gsm1k" in names


def test_filter_by_method():
    reg = load_registry()
    oren_applicable = reg.filter_by_method("oren")
    assert len(oren_applicable) > 0
    for spec in oren_applicable:
        assert "oren" in spec.applicable_methods
