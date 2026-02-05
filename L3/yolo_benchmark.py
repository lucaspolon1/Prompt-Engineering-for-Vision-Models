
import time
import torch
from ultralytics import YOLO
import argparse
from benchmark_helpers import run_yolo, run_yolo_cpu

# something about running 5 warmups and averaging 100 trials to get steady state benchmarks.
# personal bash: salloc --gres=gpu:1 --mem=16G --time=00:10:00, nvidia-smi

# CONSTRAINTS
NUM_WARMUPS = 5
NUM_RUNS = 99
OUTPUT_ROOT = "outputs/yolo_test6" # single folder for each run
YOLO_MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
]

# TODO

# one run should output m folders each having n pictures
# right now each picture goes into a different folder. I'd like them to all go in the same folder
# labeling system for photos


def main(device):
    for model_name in YOLO_MODELS:
        print(f"\nLoading {model_name}...")
        model = YOLO(model_name)

        if device == "gpu":
            model.to("cuda")
        else:
            model.to("cpu")
        
        # WARMUP RUNS (5 for now) to achieve steady state data collection
        for i in range(5):
            run_yolo(model, device, save_outputs=False)

        # --- RECORDED RUN
        runtimes = []

        for i in range(NUM_RUNS):
            run_id = f"{i+1:02d}"
            out_name = f"bus_{model_name.replace('.pt','')}{run_id}"

            _, runtime = run_yolo(
                model,
                device,
                output_dir="outputs/yolo_test5",
                save_outputs=True,
                out_name=out_name
            )
            runtimes.append(runtime)

        avg_time = sum(runtimes) / len(runtimes) #calculate avg time of each model.
        print(
            f"{model_name} ({device.upper()}): "
            f"avg = {avg_time:.4f} sec over {NUM_RUNS} runs"
        )
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "device",
        choices=["cpu", "gpu"], # python benchmark_yolo.py "___"
        help="Run YOLO benchmark on CPU or GPU"
    )
    args = parser.parse_args()

    main(args.device)
