"""
TSV 实验记录管理
负责实验历史的初始化、写入和读取
"""

import csv
from typing import Dict, List, Any

from config.settings import TSV_PATH


def init_tsv():
    """初始化 TSV 文件（如果不存在）"""
    import os
    if not os.path.exists(TSV_PATH):
        with open(TSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["commit", "map50", "map50_95", "status", "description"])
        print(f"[初始化] 创建 {TSV_PATH}")


def log_tsv(commit: str, map50: float, map50_95: float, status: str, description: str):
    """记录一行实验结果到 TSV"""
    with open(TSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            commit,
            f"{map50:.6f}",
            f"{map50_95:.6f}",
            status,
            description
        ])


def load_tsv_history() -> List[Dict[str, Any]]:
    """从 TSV 加载历史记录。

    容错：手工编辑或中断写坏的行会被跳过并给出提示，
    而不是让整个 Agent 在启动时崩掉。
    """
    history = []
    import os
    if not os.path.exists(TSV_PATH):
        return history

    def _f(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    with open(TSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for lineno, row in enumerate(reader, start=2):
            raw = row.get("map50")
            try:
                map50 = float(raw) if raw not in (None, "") else 0.0
            except (TypeError, ValueError):
                print(f"[历史] 跳过第 {lineno} 行：map50 无法解析 ({raw!r})")
                continue
            history.append({
                "commit": row.get("commit", "") or "",
                "map50": map50,
                "map50_95": _f(row.get("map50_95")),
                "status": row.get("status", "") or "",
                "description": row.get("description", "") or "",
            })
    return history
