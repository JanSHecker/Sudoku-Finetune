from __future__ import annotations

import argparse
import json
from pathlib import Path

from peft import PeftModel

from train import OUTPUT_DIR, build_system_prompt, generate_labels, load_base_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one banking support request.")
    parser.add_argument("text", nargs="+", help="The support request to classify")
    parser.add_argument("--adapter", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    metrics_path = args.adapter / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"Missing {metrics_path}; train and save the adapter first")

    labels = json.loads(metrics_path.read_text(encoding="utf-8"))["labels"]
    base_model, tokenizer = load_base_model()
    model = PeftModel.from_pretrained(base_model, str(args.adapter))
    prediction = generate_labels(
        model,
        tokenizer,
        [" ".join(args.text)],
        build_system_prompt(labels),
        batch_size=1,
    )[0]
    if prediction not in labels:
        raise SystemExit(f"Model returned an invalid label: {prediction!r}")
    print(prediction)


if __name__ == "__main__":
    main()
