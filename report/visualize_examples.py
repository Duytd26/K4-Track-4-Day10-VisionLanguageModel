"""Draw six real test predictions alongside ground truth; run after exporting JSONL."""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.labcommon import read_jsonl  # noqa: E402
from scripts.score import _gt_boxes, _match, _to_norm1000, parse_output  # noqa: E402


def _pixel_box(box, width, height):
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, round(x1 * width / 1000))),
        max(0, min(height - 1, round(y1 * height / 1000))),
        max(0, min(width - 1, round(x2 * width / 1000))),
        max(0, min(height - 1, round(y2 * height / 1000))),
    )


def _draw_box(draw, item, color, width, height):
    x1, y1, x2, y2 = _pixel_box(item["bbox"], width, height)
    if x2 <= x1 or y2 <= y1:
        return
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    label_y = y1 if y1 >= 13 else min(height - 12, y2 + 2)
    text = item["label"]
    draw.rectangle((x1, label_y, min(width - 1, x1 + len(text) * 7 + 4), label_y + 12), fill=color)
    draw.text((x1 + 2, label_y), text, fill="white")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, help="work/e1.jsonl or another real prediction file")
    parser.add_argument("--split", default="data/splits/test.jsonl")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--coord", choices=("norm1000", "pixel"), default="norm1000")
    parser.add_argument("--out-dir", default="report/examples")
    args = parser.parse_args()

    entries = read_jsonl(args.split)
    prediction_rows = read_jsonl(args.pred)
    preds = {row["image_id"]: row["raw"] for row in prediction_rows}
    if len(preds) != len(prediction_rows):
        raise ValueError("Duplicate image_id in prediction file")
    missing = [row["image_id"] for row in entries if row["image_id"] not in preds]
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} images")

    cases = []
    for row in entries:
        parsed = parse_output(preds[row["image_id"]])
        predicted = _to_norm1000(parsed or [], args.coord, row["width"], row["height"])
        gt = _gt_boxes(row)
        per_class, _ = _match(gt, predicted)
        tp, fp, fn = (sum(counts[i] for counts in per_class.values()) for i in range(3))
        cases.append((row, gt, predicted, parsed is not None, tp, fp, fn))

    correct = sorted((c for c in cases if c[4] > 0), key=lambda c: (-c[4], c[5] + c[6], c[0]["image_id"]))[:3]
    correct_ids = {c[0]["image_id"] for c in correct}
    errors = sorted(
        (c for c in cases if c[5] + c[6] > 0 and c[0]["image_id"] not in correct_ids),
        key=lambda c: (-(c[5] + c[6]), c[4], c[0]["image_id"]),
    )[:3]
    if len(correct) < 3 or len(errors) < 3:
        raise ValueError("Need at least three images with a true positive and three with an error")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = ["# Sáu ví dụ thật", "", "Xanh lá: ground truth; đỏ: dự đoán. TP/FP/FN tính theo IoU≥0,5 và cùng lớp.", ""]
    for group, selected in (("correct", correct), ("error", errors)):
        for index, (row, gt, predicted, valid, tp, fp, fn) in enumerate(selected, 1):
            image_path = Path(args.data_root) / row["image_path"]
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            draw = ImageDraw.Draw(image)
            for item in gt:
                _draw_box(draw, item, "green", *image.size)
            for item in predicted:
                _draw_box(draw, item, "red", *image.size)
            filename = f"{group}_{index:02d}_{row['image_id']}.png"
            image.save(out_dir / filename)
            summary.append(
                f"- [{filename}]({filename}): `{row['image_id']}`, "
                f"TP={tp}, FP={fp}, FN={fn}, JSON={'hợp lệ' if valid else 'không hợp lệ'}."
            )
    (out_dir / "README.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Wrote 6 real examples to {out_dir}")


if __name__ == "__main__":
    main()
