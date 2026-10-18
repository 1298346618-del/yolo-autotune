"""
monitor.py — 训练监控仪表盘

用法：
    python monitor.py                       # 每 3 秒刷新
    python monitor.py --log tune_log.txt    # 指定日志文件
    python monitor.py --once                # 只打印一次（适合放进脚本/日志）

显示：当前轮次与 epoch 进度、最近日志、results.tsv 表格、当前最佳。

配套用法（把 main.py 的输出同时写进日志）：
    python main.py 2>&1 | tee tune_log.txt      # bash
    python main.py *>&1 | Tee-Object tune_log.txt   # PowerShell 7+
"""

import argparse
import os
import re
import sys
import time


ANSI_RESET = "\033[0m"
ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_YELLOW = "\033[33m"
ANSI_CYAN = "\033[36m"
ANSI_BOLD = "\033[1m"


def _enable_ansi_on_windows():
    """Windows 10+ 终端默认不开 VT 转义，这里打开它（失败则静默跳过）。"""
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass


def read_log(path: str) -> str:
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def parse_round(log: str):
    m = re.findall(r"第 (\d+)/(\d+) 轮实验", log)
    if not m:
        return None, None
    cur, total = m[-1]
    return int(cur), int(total)


def parse_epoch(log: str):
    m = re.findall(r"^\s*(\d+)/(\d+)\s", log, re.MULTILINE)
    if not m:
        return None, None
    e, total = m[-1]
    return int(e), int(total)


def parse_status(log: str) -> str:
    if "训练失败" in log or "Traceback" in log:
        return "error"
    return "running"


def read_tsv(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().strip().splitlines()
    except OSError:
        return []
    rows = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) >= 5:
            rows.append(parts)
    return rows


def fmt_bar(ratio: float, width: int = 24) -> str:
    ratio = max(0.0, min(1.0, ratio))
    filled = int(round(ratio * width))
    return "█" * filled + "░" * (width - filled)


def render(log_path: str, tsv_path: str) -> str:
    log = read_log(log_path)
    out = []

    out.append(ANSI_BOLD + "YOLO Auto-Tune 监控仪表盘" + ANSI_RESET + "  (Ctrl+C 退出)")

    cur_r, total_r = parse_round(log)
    cur_e, total_e = parse_epoch(log)

    if not log:
        state = ANSI_YELLOW + "[等待启动...]" + ANSI_RESET
    elif "实验结束" in log or "最佳权重保存在" in log:
        state = ANSI_GREEN + "[已完成]" + ANSI_RESET
    elif parse_status(log) == "error":
        state = ANSI_RED + "[出错]" + ANSI_RESET
    else:
        state = ANSI_CYAN + "[训练中]" + ANSI_RESET

    out.append(f"状态: {state}")
    out.append(f"轮次: {cur_r}/{total_r} {fmt_bar(cur_r / total_r)}" if cur_r else "轮次: -")
    out.append(f"epoch: {cur_e}/{total_e} {fmt_bar(cur_e / total_e)}" if cur_e else "epoch: -")

    tail_lines = [l for l in log.splitlines() if l.strip()][-4:]
    tail = " | ".join(l.strip() for l in tail_lines)
    if len(tail) > 100:
        tail = tail[-100:]
    out.append("")
    out.append(ANSI_CYAN + "最近日志: " + ANSI_RESET + tail)

    rows = read_tsv(tsv_path)
    out.append("")
    out.append(ANSI_BOLD + f"实验结果 ({os.path.basename(tsv_path)})" + ANSI_RESET)
    if rows:
        out.append(f"{'轮次':<10}{'mAP50':<12}{'mAP50-95':<12}{'状态':<10}描述")
        out.append("-" * 72)
        best = None
        for r in rows:
            commit, map50, map50_95, status_v, desc = r[:5]
            try:
                m50 = float(map50)
            except ValueError:
                m50 = 0.0
            if best is None or m50 > best[0]:
                best = (m50, r)
            color = ANSI_GREEN if status_v == "keep" else (ANSI_RED if status_v == "crash" else "")
            out.append(f"{commit:<10}{map50:<12}{map50_95:<12}{color}{status_v:<10}{ANSI_RESET}{desc[:32]}")
        if best:
            out.append("")
            out.append(ANSI_BOLD + ANSI_GREEN +
                       f"当前最佳: {best[1][0]}  mAP50={best[1][1]}  ({best[1][4][:40]})" +
                       ANSI_RESET)
    else:
        out.append("(暂无实验记录)")

    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description="YOLO Auto-Tune 训练监控")
    p.add_argument("--log", default="tune_log.txt", help="main.py 的输出日志")
    p.add_argument("--tsv", default="results.tsv", help="实验记录 TSV")
    p.add_argument("--refresh", type=float, default=3.0, help="刷新间隔（秒）")
    p.add_argument("--once", action="store_true", help="只打印一次")
    args = p.parse_args()

    _enable_ansi_on_windows()
    log_path = os.path.abspath(args.log)
    tsv_path = os.path.abspath(args.tsv)

    if args.once:
        print(render(log_path, tsv_path))
        return

    try:
        while True:
            print("\033[2J\033[H" + render(log_path, tsv_path))
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\n监控已退出")


if __name__ == "__main__":
    main()
