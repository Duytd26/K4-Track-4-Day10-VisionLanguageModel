"""Reproduce the E3/E5 paired comparison and E5 reward-hacking audit.

Run ``python report/e5_audit.py`` after ``scripts/prepare_data.py`` creates
``data/splits/test.jsonl``. Predictions and the GRPO log are tracked under
``report/``; dataset images and labels remain ignored by Git.
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes, _match, _prf, iou_xyxy, parse_output


def load_predictions(path):
    rows = read_jsonl(path)
    return {row["image_id"]: row["raw"] for row in rows}, len(rows)


def counts(entry, raw):
    boxes = parse_output(raw)
    if boxes is None:
        boxes = []
    per_class, _ = _match(_gt_boxes(entry), boxes)
    return tuple(sum(values[i] for values in per_class.values()) for i in range(3))


def f1(rows):
    return _prf(*(sum(row[i] for row in rows) for i in range(3)))["f1"]


entries = read_jsonl(ROOT / "data/splits/test.jsonl")
pred3, n3 = load_predictions(ROOT / "report/predictions/e3.jsonl")
pred5, n5 = load_predictions(ROOT / "report/predictions/e5.jsonl")
ids = {entry["image_id"] for entry in entries}
assert len(entries) == 311 and n3 == n5 == len(ids) == 311
assert set(pred3) == set(pred5) == ids

result = {"n_images": len(entries)}
model_rows = {}
for name, predictions in (("E3", pred3), ("E5", pred5)):
    model_rows[name] = [counts(entry, predictions[entry["image_id"]]) for entry in entries]
    empty = empty_with_gt = giant = out_of_bounds = duplicate_pairs = 0
    classes = Counter()
    sessions = defaultdict(list)
    for entry, row in zip(entries, model_rows[name]):
        boxes = parse_output(predictions[entry["image_id"]])
        assert boxes is not None
        if not boxes:
            empty += 1
            empty_with_gt += bool(entry["objects"])
        for box in boxes:
            classes[box["label"]] += 1
            x1, y1, x2, y2 = box["bbox"]
            giant += (x2 - x1) * (y2 - y1) >= 500_000
            out_of_bounds += min(x1, y1) < 0 or max(x2, y2) > 1000
        for i, a in enumerate(boxes):
            for b in boxes[i + 1 :]:
                duplicate_pairs += a["label"] == b["label"] and iou_xyxy(a["bbox"], b["bbox"]) >= 0.9
        sessions[entry["session"]].append(row)
    result[name] = {
        "f1_recomputed": f1(model_rows[name]),
        "empty_outputs": empty,
        "empty_outputs_with_gt": empty_with_gt,
        "giant_boxes_area_ge_half_image": giant,
        "out_of_bounds_boxes": out_of_bounds,
        "duplicate_box_pairs_iou_ge_0_9": duplicate_pairs,
        "predicted_boxes_by_class": dict(classes),
        "sessions": {s: {"n_images": len(rows), "f1": f1(rows)} for s, rows in sessions.items()},
    }

rng = random.Random(42)
deltas = []
for _ in range(5000):
    sampled = [rng.randrange(len(entries)) for _ in entries]
    deltas.append(f1([model_rows["E5"][i] for i in sampled]) - f1([model_rows["E3"][i] for i in sampled]))
deltas.sort()
result["paired_bootstrap_delta_e5_minus_e3"] = {
    "point": f1(model_rows["E5"]) - f1(model_rows["E3"]),
    "ci95": [deltas[125], deltas[4874]],
    "n_boot": len(deltas),
    "seed": 42,
    "sampling_unit": "image",
}

log = json.loads((ROOT / "report/training/e5_trainer_log.json").read_text(encoding="utf-8"))
steps = [row for row in log if "reward" in row]
assert len(steps) == 60 and [row["step"] for row in steps] == list(range(1, 61))
result["grpo_log"] = {
    "logged_steps": len(steps),
    "reward_mean_first_10": sum(row["reward"] for row in steps[:10]) / 10,
    "reward_mean_last_10": sum(row["reward"] for row in steps[-10:]) / 10,
    "reward_mean_all_steps": sum(row["reward"] for row in steps) / len(steps),
    "zero_reward_std_steps": sum(row["reward_std"] == 0 for row in steps),
    "zero_grad_norm_steps": sum(row["grad_norm"] == 0 for row in steps),
    "last_reward": steps[-1]["reward"],
    "last_reward_std": steps[-1]["reward_std"],
}
print(json.dumps(result, ensure_ascii=False, indent=2))
