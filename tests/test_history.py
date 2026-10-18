"""TSV 历史记录的读写在坏数据下也不应该崩。"""

import core.history as history


def test_init_creates_header(tmp_path, monkeypatch):
    tsv = tmp_path / "results.tsv"
    monkeypatch.setattr(history, "TSV_PATH", str(tsv))
    history.init_tsv()
    assert tsv.exists()
    assert tsv.read_text(encoding="utf-8").strip().split("\t")[:2] == ["commit", "map50"]


def test_log_then_reload(tmp_path, monkeypatch):
    tsv = tmp_path / "results.tsv"
    monkeypatch.setattr(history, "TSV_PATH", str(tsv))
    history.init_tsv()
    history.log_tsv("iter0", 0.783617, 0.555783, "keep", "增大 box 至 15.0")

    rows = history.load_tsv_history()
    assert len(rows) == 1
    assert rows[0]["commit"] == "iter0"
    assert abs(rows[0]["map50"] - 0.783617) < 1e-9
    assert rows[0]["status"] == "keep"


def test_malformed_row_is_skipped(tmp_path, monkeypatch):
    tsv = tmp_path / "results.tsv"
    tsv.write_text(
        "commit\tmap50\tmap50_95\tstatus\tdescription\n"
        "iter0\t0.8\t0.5\tkeep\tok\n"
        "iter1\tnot-a-number\t0.5\tdiscard\tbroken\n"
        "iter2\t0.82\t0.51\tkeep\tok\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(history, "TSV_PATH", str(tsv))
    rows = history.load_tsv_history()
    assert [r["commit"] for r in rows] == ["iter0", "iter2"]


def test_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "TSV_PATH", str(tmp_path / "nope.tsv"))
    assert history.load_tsv_history() == []
