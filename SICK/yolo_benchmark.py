
import time
import torch
from ultralytics import YOLO
import argparse
from benchmark_helpers import run_yolo, get_model_accuracy

# personal bash: salloc --gres=gpu:1 --mem=16G --time=00:10:00, nvidia-smi

# CONSTRAINTS
NUM_WARMUPS = 5
NUM_RUNS = 20
OUTPUT_ROOT = "outputs/yolo_test7" # single folder for each run
YOLO_MODELS = [
    # "yolov8n.pt",
    # "yolov8s.pt",
    # "yolov8m.pt",
    # "yolov8l.pt",
    # "yolov8x.pt",

    # "yolo11n.pt",  # YOLOv11 nano
    # "yolo11s.pt",  # YOLOv11 small
    # "yolo11m.pt",  # YOLOv11 medium
    # "yolo11l.pt",  # YOLOv11 large
    # "yolo11x.pt",  # YOLOv11 xlarge

    # "yolov10n.pt", # speed optimized models
    # "yolov10s.pt",
    # "yolov10m.pt",
    # "yolov10l.pt",
    # "yolov10x.pt",

    # "yolov9t.pt",  # tiny
    # "yolov9s.pt",
    # "yolov9m.pt",
    # "yolov9c.pt",  # compact
    # "yolov9e.pt",  # extended
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26l.pt",
    "yolo26x.pt",
]
MAP_METRIC = "mAP50" # "mAP50-95", "mAP50", "mAP75"

# TODO
# document everything
# gpu data (helpful to save data locally between every model
# maybe put graphing functionality on a different code)

# adjust data set (too big and not targeted for detection trash in nature)

# watch L4
# think about combining SAM with LLM (maybe custom ChatGPT) 
        # -- probably more interested in SLM


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
                output_dir=OUTPUT_ROOT,
                save_outputs=True,
                out_name=out_name
            )
            runtimes.append(runtime)

        avg_time = sum(runtimes) / len(runtimes)

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
    print(f"SUMMARY (using {MAP_METRIC} for optimization)")
    print(f"{'='*60}")
    print(f"{'Model':<12} {MAP_METRIC:<10} {'Avg Time (s)':<15}")
    print("-" * 60)
    for r in all_results: 
        print(f"{r['model']:<12} {r[MAP_METRIC]:<10.3f} {r['avg_runtime_sec']:<15.4f}")

    # Save results to CSV
    from benchmark_helpers import save_results_to_csv, plot_speed_accuracy_tradeoff
    save_results_to_csv(all_results, f"yolo_benchmark_{device}.csv")
    
    # Tradeoff plot (pass which metric to use)
    plot_speed_accuracy_tradeoff(all_results, f"yolo_tradeoff_{device}.png", map_metric=MAP_METRIC)
    
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
