import pytest
from app.config import load_config
from app.service import Service
from app.embedder import normalize, HashEmbedder
from app.cache import is_cacheable
from app import router


@pytest.fixture
def svc(tmp_path):
    cfg = load_config(db_path=str(tmp_path / "t.db"), llm_mode="mock", embedder_kind="hash",
                      mock_latency_scale=0.0, threshold=0.9)
    return Service(cfg)


def test_normalize():
    assert normalize("  How do I RESET, my password?! ") == "how do i reset my password"


def test_exact_repeat_hits_cache_and_is_free(svc):
    a = svc.answer("How do I reset my password?")
    b = svc.answer("how do i reset my password")
    assert not a["cache_hit"] and b["cache_hit"] and b["cost"] == 0.0 and a["answer"] == b["answer"]


def test_corpus_version_bump_invalidates(svc):
    svc.answer("How do I reset my password?")
    svc.bump_corpus_version()
    assert not svc.answer("How do I reset my password?")["cache_hit"]


def test_personalized_never_cached(svc):
    svc.answer("How do I reset my password?", personalized=True)
    assert len(svc.cache) == 0
    assert not svc.answer("How do I reset my password?", personalized=True)["cache_hit"]


def test_tenant_isolation(svc):
    svc.answer("How do I reset my password?", tenant="acme")
    assert not svc.answer("How do I reset my password?", tenant="globex")["cache_hit"]
    assert svc.answer("How do I reset my password?", tenant="acme")["cache_hit"]


def test_never_cache_bad_results():
    assert not is_cacheable({"answer": "", "error": None}, False)
    assert not is_cacheable({"answer": "x", "error": "boom"}, False)
    assert not is_cacheable({"answer": "I don't know.", "error": None}, False)
    assert not is_cacheable({"answer": "fine", "error": None}, True)


def test_ttl_expiry(svc):
    svc.cache.ttl = -1
    svc.answer("How do I reset my password?")
    assert not svc.answer("How do I reset my password?")["cache_hit"]


def test_escalation_on_hard_query_keeps_quality(svc):
    cfg = svc.cfg; cfg.complex_cutoff = 99  # force everything to "simple" so escalation must rescue it
    r = svc.answer("Why does the EU refund policy differ from the US one?", expected=["14 days", "30 days"])
    assert r["escalated"] and r["model"] == cfg.large_model


def test_router_rules():
    cfg = load_config()
    assert router.classify_rules("Compare the Free and Pro plans", [0.5, 0.2, 0.1], cfg) == "complex"
    assert router.classify_rules("How much is Pro?", [0.5, 0.2, 0.1], cfg) == "simple"
