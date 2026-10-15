"""Resume the in-memory E5 trainer after TRL's quantized-model BF16 cast.

Run from a Colab cell with ``resume_e5(globals())`` after the first-step
Half/BFloat16 LayerNorm error. It reuses the model already in RAM for training;
test inference loads the saved result after training completes.
"""

import gc
import json
import shutil
import time
from pathlib import Path

import torch


def resume_e5(context):
    required = (
        "e5_trainer", "E5_DIR", "e5_cfg", "source_run", "E5_SOURCE_SETUP",
        "e5_dataset_version", "trainable_params_e5", "PROJECTOR_MARKERS",
        "DRIVE_RUNS", "infer_and_score", "e5_baselines",
    )
    missing = [name for name in required if name not in context]
    if missing:
        raise RuntimeError(f"E5 setup cell must run first; missing: {missing}")

    trainer = context["e5_trainer"]
    if trainer.state.global_step != 0:
        raise RuntimeError(
            f"Trainer already completed {trainer.state.global_step} steps; "
            "do not restart it with this first-step recovery helper."
        )

    # GRPOTrainer 1.9.2 casts all trainable parameters of a quantized model
    # to BF16 in its constructor, even with fp16=True. Restore the SFT setup
    # after construction so projector LayerNorm and GradScaler can use FP32.
    for param in trainer.model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    trainable_dtypes = {
        param.dtype for param in trainer.model.parameters() if param.requires_grad
    }
    if trainable_dtypes != {torch.float32}:
        raise RuntimeError(f"Trainable dtypes after repair: {trainable_dtypes}")
    trainer.model.zero_grad(set_to_none=True)
    if trainer.optimizer is not None and trainer.optimizer.state:
        raise RuntimeError("Optimizer has nonempty state; first-step recovery is unsafe")

    print("E5 trainable parameters restored to FP32; resuming trainer in RAM with FP16 autocast.")
    torch.cuda.reset_peak_memory_stats()
    train_started = time.perf_counter()
    # GRPO's rollout generation is outside Trainer's mixed-precision forward
    # context. The E3 visual backbone is FP16 while its trainable projector is
    # FP32; autocast must cover both rollout generation and policy updates.
    with torch.autocast("cuda", dtype=torch.float16):
        train_result = trainer.train()
    train_minutes = (time.perf_counter() - train_started) / 60.0
    peak_vram_gb = torch.cuda.max_memory_allocated(0) / (1024**3)

    e5_dir = Path(context["E5_DIR"])
    checkpoint = e5_dir / "checkpoint-final"
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter))
    source_run = context["source_run"]
    projector_params = int(source_run.get("projector_params", 0))
    if projector_params:
        markers = context["PROJECTOR_MARKERS"]
        projector_state = {
            name: param.detach().cpu().contiguous()
            for name, param in trainer.model.named_parameters()
            if any(marker in name for marker in markers)
        }
        if sum(param.numel() for param in projector_state.values()) != projector_params:
            raise RuntimeError("Projector parameter count changed during E5")
        torch.save(projector_state, checkpoint / "projector.pt")

    config = context["e5_cfg"]
    source_setup = context["E5_SOURCE_SETUP"]
    run_info = {
        "name": "E5",
        "base_setup": source_setup,
        "source_setup": source_setup,
        "model_id": source_run["model_id"],
        "lora_r": int(source_run["lora_r"]),
        "lora_alpha": int(source_run["lora_alpha"]),
        "projector_params": projector_params,
        "trainable_params": int(context["trainable_params_e5"]),
        "train_minutes": float(train_minutes),
        "peak_vram_gb": float(peak_vram_gb),
        "learning_rate": float(config["learning_rate"]),
        "num_generations": int(config["num_generations"]),
        "max_steps": int(config["max_steps"]),
        "beta": float(config["beta"]),
        "completed_steps": int(trainer.state.global_step),
        "dataset_version": context["e5_dataset_version"],
        "seed": int(config.get("seed", 42)),
        "checkpoint": str(checkpoint.resolve()),
    }
    (checkpoint / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    (e5_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    (e5_dir / "trainer_log.json").write_text(
        json.dumps(trainer.state.log_history, indent=2), encoding="utf-8"
    )

    drive_dir = Path(context["DRIVE_RUNS"]) / "E5"
    drive_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(checkpoint, drive_dir / checkpoint.name, dirs_exist_ok=True)
    shutil.copy2(e5_dir / "run.json", drive_dir / "run.json")
    shutil.copy2(e5_dir / "trainer_log.json", drive_dir / "trainer_log.json")
    print(f"E5 checkpoint and log backed up to {drive_dir}")

    context["run_info_e5"] = run_info
    context["e5_train_result"] = train_result
    context.pop("e5_trainer", None)
    context.pop("e5_model", None)
    context.pop("grpo_dataset", None)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    evidence = context["infer_and_score"]("E5", evidence_name="E5", coordinate="norm1000")
    context["e5_evidence"] = evidence
    sft = context["e5_baselines"][source_setup]
    sft_ci = sft["f1_ci95"]
    e5_ci = evidence["f1_ci95"]
    summary = {
        "evidence": "report/evidence/e5.json",
        "source_setup": source_setup,
        "source_sft_f1": sft["micro"]["f1"],
        "e5_f1": evidence["micro"]["f1"],
        "f1_delta": evidence["micro"]["f1"] - sft["micro"]["f1"],
        "sft_ci95": sft_ci,
        "e5_ci95": e5_ci,
        "ci95_overlap_descriptive_only": max(sft_ci[0], e5_ci[0]) <= min(sft_ci[1], e5_ci[1]),
        "train_minutes": train_minutes,
        "peak_vram_gb": peak_vram_gb,
        "completed_steps": run_info["completed_steps"],
        "trainer_log": str(e5_dir / "trainer_log.json"),
    }
    print(json.dumps(summary, indent=2))
    return summary
