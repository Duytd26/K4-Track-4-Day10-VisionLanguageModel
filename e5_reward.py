"""Verifiable IoU reward for the optional Day 10 GRPO experiment."""

from scripts.score import iou_xyxy, parse_output


def compute_reward(completion: str, gt_boxes: list[dict]) -> float:
    """Score valid JSON and greedily matched same-class boxes at IoU >= 0.5."""
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0

    total_iou = 0.0
    all_labels = {box["label"] for box in gt_boxes} | {box["label"] for box in pred}
    for label in all_labels:
        gt = [box["bbox"] for box in gt_boxes if box["label"] == label]
        predicted = [box["bbox"] for box in pred if box["label"] == label]
        pairs = sorted(
            [
                (iou_xyxy(a, b), i, j)
                for i, a in enumerate(gt)
                for j, b in enumerate(predicted)
            ],
            reverse=True,
        )
        used_gt, used_pred = set(), set()
        for overlap, i, j in pairs:
            if overlap < 0.5:
                break
            if i in used_gt or j in used_pred:
                continue
            used_gt.add(i)
            used_pred.add(j)
            total_iou += overlap

    denominator = max(len(gt_boxes), len(pred), 1)
    return 0.2 + 0.8 * total_iou / denominator


def completion_text(completion) -> str:
    """Extract generated text from TRL standard or conversational completion output."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        for message in reversed(completion):
            if isinstance(message, dict) and "content" in message:
                return str(message["content"])
    return ""


def verifiable_iou_reward(completions, gt_boxes, **kwargs) -> list[float]:
    """TRL custom reward callback; dataset columns are repeated per generated completion."""
    return [
        compute_reward(completion_text(completion), target)
        for completion, target in zip(completions, gt_boxes, strict=True)
    ]


def run_reward_unit_checks() -> dict[str, float]:
    """Run five lightweight CPU checks before starting optional GRPO training."""
    box = {"label": "pothole", "bbox": (100.0, 100.0, 300.0, 300.0)}
    exact = '[{"label":"pothole","bbox_2d":[100,100,300,300]}]'
    wrong_class = '[{"label":"crack","bbox_2d":[100,100,300,300]}]'
    extra_box = (
        '[{"label":"pothole","bbox_2d":[100,100,300,300]},'
        '{"label":"pothole","bbox_2d":[600,600,800,800]}]'
    )
    results = {
        "perfect": compute_reward(exact, [box]),
        "invalid_json": compute_reward("not json", [box]),
        "wrong_class": compute_reward(wrong_class, [box]),
        "extra_box": compute_reward(extra_box, [box]),
        "both_empty": compute_reward("[]", []),
    }
    expected = {
        "perfect": 1.0,
        "invalid_json": 0.0,
        "wrong_class": 0.2,
        "extra_box": 0.6,
        "both_empty": 1.0,
    }
    for name, value in results.items():
        assert abs(value - expected[name]) < 1e-9, (name, value, expected[name])
    return results
