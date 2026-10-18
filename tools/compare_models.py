"""
compare_models.py — 对比"调优前 / 调优后"两个权重在同一验证集上的表现

用法：
    python tools/compare_models.py \
        --data data.yaml \
        --original path/to/original.pt \
        --tuned path/to/tuned.pt

输出整体指标差异，以及逐类 mAP50（ultralytics 提供时）。
这就是 docs/case_study_pcb.md 里那张对比表的来源。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate(model_path: str, data: str, imgsz: int, device, batch: int, name: str):
    from ultralytics import YOLO

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"权重不存在: {model_path}")

    model = YOLO(model_path)
    res = model.val(
        data=data,
        imgsz=imgsz,
        device=device,
        batch=batch,
        project="compare_runs",
        name=name,
        exist_ok=True,
        verbose=False,
        plots=False,
    )

    box = res.box
    overall = {
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "Precision": float(box.mp),
        "Recall": float(box.mr),
    }
    per_class = {}
    try:
        names = res.names
        ap50 = list(box.ap50) if hasattr(box, "ap50") else []
        if ap50 and len(ap50) == len(names):
            for idx, cls_name in names.items():
                per_class[cls_name] = float(ap50[idx])
    except Exception:
        pass
    return overall, per_class


def pct(new: float, old: float) -> str:
    if old == 0:
        return "—"
    return f"{(new - old) / old * 100:+.1f}%"


def main():
    p = argparse.ArgumentParser(description="对比调优前后的 YOLO 权重")
    p.add_argument("--data", required=True, help="数据集 yaml")
    p.add_argument("--original", required=True, help="调优前的权重")
    p.add_argument("--tuned", required=True, help="调优后的权重")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0", help="0 / cpu")
    args = p.parse_args()

    print(f"[1/2] 评估原始模型: {args.original}")
    base, base_cls = evaluate(args.original, args.data, args.imgsz, args.device, args.batch, "original")
    print(f"[2/2] 评估调优模型: {args.tuned}")
    tuned, tuned_cls = evaluate(args.tuned, args.data, args.imgsz, args.device, args.batch, "tuned")

    print("\n" + "=" * 64)
    print(f"{'指标':<14}{'调优前':>12}{'调优后':>12}{'变化':>12}")
    print("-" * 64)
    for k in ["mAP50", "mAP50-95", "Precision", "Recall"]:
        print(f"{k:<14}{base[k]:>12.4f}{tuned[k]:>12.4f}{pct(tuned[k], base[k]):>12}")
    print("=" * 64)

    if base_cls and tuned_cls:
        print(f"\n{'类别':<20}{'调优前':>10}{'调优后':>10}{'变化':>10}")
        print("-" * 52)
        for cls in base_cls:
            if cls in tuned_cls:
                print(f"{cls:<20}{base_cls[cls]:>10.4f}{tuned_cls[cls]:>10.4f}"
                      f"{pct(tuned_cls[cls], base_cls[cls]):>10}")


if __name__ == "__main__":
    main()
