import time
import torch
from ultralytics import YOLO
import os
import cv2

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

def get_model_accuracy(model, model_name):
    """Run validation to get mAP metrics.
    Returns dict with mAP scores.

    Args:
        model (_type_): the model
        model_name (_type_): the name of the model
    """
    print(f"Running validation for {model_name}...")

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