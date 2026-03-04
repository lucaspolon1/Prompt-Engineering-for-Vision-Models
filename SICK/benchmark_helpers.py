import time
import torch
from ultralytics import YOLO
import os
import cv2
import csv
import matplotlib.pyplot as plt

#img = "https://ultralytics.com/images/bus.jpg" # Changed to clean up output
# bash: wget https://ultralytics.com/images/bus.jpg
img = "bus.jpg"

def run_yolo(model, device, output_dir=None, save_outputs=True, out_name=None):
    start = time.time()
    results = model(img, save=False, verbose=False)

    if save_outputs and output_dir and out_name:
        os.makedirs(output_dir, exist_ok=True)
        img_out = results[0].plot()
        cv2.imwrite(
            os.path.join(output_dir, f"{out_name}.jpg"),
            img_out
        )

    return results, time.time() - start


def plot_speed_accuracy_tradeoff(results, output_filename="speed_accuracy_tradeoff.png", map_metric="mAP50-95"):
    """
    Create a scatter plot showing speed vs accuracy tradeoff.
    
    Args:
        results: List of result dictionaries
        output_filename: Where to save the plot
        map_metric: Which mAP metric to plot (e.g., "mAP50", "mAP50-95", "mAP75")
    """
    if not results:
        print("No results to plot!")
        return
    
    # Extract data
    models = [r['model'].replace('.pt', '') for r in results]
    runtimes = [r['avg_runtime_sec'] for r in results]
    map_scores = [r[map_metric] for r in results]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(runtimes, map_scores, s=100, alpha=0.6)
    
    # Label each point with model name
    for i, model in enumerate(models):
        plt.annotate(model, (runtimes[i], map_scores[i]), 
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    
    plt.xlabel('Average Runtime (seconds)', fontsize=12)
    plt.ylabel(map_metric, fontsize=12)
    plt.title(f'YOLO Model Speed vs Accuracy Tradeoff ({map_metric})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add note about optimal region
    plt.text(0.02, 0.98, 'Optimal: Top-left corner\n(high accuracy, low runtime)', 
             transform=plt.gca().transAxes, fontsize=9, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {output_filename}")
    plt.close()
    

def get_model_accuracy(model, model_name):
    """
    Run validation to get mAP metrics.
    Returns dict with mAP scores.
    """
    print(f"  Running validation for {model_name}...")
    
    # Run validation on COCO dataset (YOLO will download if needed)
    metrics = model.val(data='coco.yaml', verbose=False)
    
    # Extract mAP metrics
    accuracy_results = {
        'model': model_name,
        'mAP50-95': float(metrics.box.map),    # mAP at IoU 0.5:0.95
        'mAP50': float(metrics.box.map50),      # mAP at IoU 0.5
        'mAP75': float(metrics.box.map75),      # mAP at IoU 0.75
    }
    
    print(f"  ✓ {model_name}: mAP50-95={accuracy_results['mAP50-95']:.3f}, mAP50={accuracy_results['mAP50']:.3f}")
    
    return accuracy_results

def save_results_to_csv(results, filename="benchmark_results.csv"):
    """_summary_

    Args:
        results (_type_): _description_
        filename (str, optional): _description_. Defaults to "benchmark_results.csv".
    """
    if not results:
        print("No results to save.")
        return
    
    fieldnames=results[0].keys()

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {filename}")


def run_yolo_cpu(model_name, output_dir=None, warmup=False):
    """
    This function runs yolo model on CPU. It runs five warmup runs to get rid of transient data.

    Args:
        model_name (_type_): _description_
        output_dir (_type_, optional): _description_. Defaults to None.
        warmup (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    model = YOLO(model_name).to("cpu")

    if warmup:
        _ = model(img)
        return None, None

    start = time.time()

    results = model(
        img,
        save=(output_dir is not None),
        project=output_dir,
        name=model_name.replace(".pt", "")
    )

    end = time.time()

    return results, end - start