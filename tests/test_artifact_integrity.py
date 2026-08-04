import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_final_integrity_report_passes():
    path = ROOT / "results" / "summary" / "integrity_report.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["passed"] is True


def test_result_master_declares_uniform_seeds():
    path = ROOT / "results" / "summary" / "results_master.json"
    if not path.exists():
        return
    master = json.loads(path.read_text(encoding="utf-8"))
    assert master["dimensions"]["seeds"] == list(range(30))
