"""Re-evaluate saved source transformer checkpoints on the complete UAE pool."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.final_studies_neural import (
    DEVICE,
    MODELS_DIR,
    RESULTS_DIR,
    TASKS,
    TRANSFORMERS,
    filter_task,
    load_split,
    save_result,
    transformer_loader,
    transformer_predict,
)


STUDY = "Study 1 - Zero-shot UAE"


def metric_path(task: str, display_name: str) -> Path:
    return (
        RESULTS_DIR
        / f"{STUDY}_{task}_{display_name.lower()}_metrics.json"
    )


def main() -> None:
    for model_key in ["arabert", "marbert"]:
        display_name = TRANSFORMERS[model_key]["display_name"]
        batch_size = TRANSFORMERS[model_key]["batch_size"]
        for task, config in TASKS.items():
            checkpoint = MODELS_DIR / f"{model_key}_source_{task}"
            old_path = metric_path(task, display_name)
            old = json.loads(old_path.read_text(encoding="utf-8"))

            train = filter_task(load_split(f"source_{task}_train"), task)
            validation = filter_task(
                load_split(f"source_{task}_validation"),
                task,
            )
            test = filter_task(load_split(f"uae_{task}_full"), task)
            labels = config["labels"]
            label_to_id = {label: index for index, label in enumerate(labels)}

            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint,
                local_files_only=True,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint,
                local_files_only=True,
            ).to(DEVICE)
            loader = transformer_loader(
                tokenizer,
                test,
                label_to_id,
                max_len=128,
                batch_size=batch_size * 2,
                shuffle=False,
            )
            predictions, confidence = transformer_predict(model, loader, labels)
            metrics = save_result(
                study=STUDY,
                task=task,
                model_name=display_name,
                train=train,
                validation=validation,
                test=test,
                predictions=predictions,
                confidence=confidence,
                training_history=old.get("training_history", []),
                best_epoch=int(old.get("best_epoch", 0)),
                settings={
                    **old.get("settings", {}),
                    "evaluation_manifest": f"uae_{task}_full.csv",
                    "checkpoint_reused": str(checkpoint),
                },
            )
            print(
                f"{display_name} {task}: N={len(test)} "
                f"accuracy={metrics['accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
