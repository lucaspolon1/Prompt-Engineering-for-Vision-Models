import time
import torch
from ultralytics import YOLO
import argparse
from benchmark_helpers import run_yolo, run_yolo_cpu

# something about running 5 warmups and averaging 100 trials to get steady state benchmarks.
# personal bash: salloc --gres=gpu:1 --mem=16G --time=00:10:00, nvidia-smi


YOLO_MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
]

# then I want to have it run "n" times (variable you can change at the top of the script) for each model
# where there are "m" models
# one run should output m folders each having n pictures
# currently the helper recreates the model every call.


def main(device):
    for model_name in YOLO_MODELS:
        model = YOLO(model_name)

        if device == "gpu":
            model.to("cuda")
        else:
            model.to("cpu")
        
        # WARMUP RUNS (5 for now) to achieve steady state data collection
        for i in range(5):
            run_yolo(model, device, save_outputs=False)

        # --- RECORDED RUN
        print(f"Benchmarking {model_name}...")
        _, runtime = run_yolo(
            model,
            device,
            output_dir="outputs/yolo_test2",
            save_outputs=True
        )
        
        print(f"{model_name} ({device.upper()}): {runtime:.4f} sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "device",
        choices=["cpu", "gpu"], # python benchmark_yolo.py "___"
        help="Run YOLO benchmark on CPU or GPU"
    )
    args = parser.parse_args()

    main(args.device)
