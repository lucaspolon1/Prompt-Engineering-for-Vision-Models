
import time
import torch
from ultralytics import YOLO
import argparse
from benchmark_helpers import run_yolo, get_model_accuracy

# personal bash: salloc --gres=gpu:1 --mem=16G --time=00:10:00, nvidia-smi

# CONSTRAINTS
NUM_WARMUPS = 5
NUM_RUNS = 20
OUTPUT_ROOT = "outputs/yolo_test6" # single folder for each run
YOLO_MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    #"yolov8l.pt",
    #"yolov8x.pt",
]

# TODO
# implement model validation (model.val()) 
    # model val should be once per model for mAP



def main(device):
    all_results = []

    for model_name in YOLO_MODELS:
        print(f"\n{'='*60}")
        print(f"Benchmarking {model_name}")
        print(f"{'='*60}")

        model = YOLO(model_name)

        # --- ACCURACY (new)
        print("\n[PHASE 1: Accuracy Measurement]")
        accuracy_results = get_model_accuracy(model, model_name)

        # --- SPEED
        print(f"\n[PHASE 2: Speed Measurement on {device.upper()}]")

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

        # Combine results
        result = {
            **accuracy_results,
            'device': device,
            'avg_runtime_sec': avg_time,
            'num_runs': NUM_RUNS
        }
        all_results.append(result)

        print(
            f"{model_name} ({device.upper()}): "
            f"Average runtime: {avg_time:.4f} sec"
        )

    # --- SUMMARY TABLE ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'mAP50-95':<10} {'mAP50':<10} {'Avg Time (s)':<15}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['model']:<12} {r['mAP50-95']:<10.3f} {r['mAP50']:<10.3f} {r['avg_runtime_sec']:<15.4f}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "device",
        choices=["cpu", "gpu"], # python benchmark_yolo.py "___"
        help="Run YOLO benchmark on CPU or GPU"
    )
    args = parser.parse_args()

    main(args.device)
