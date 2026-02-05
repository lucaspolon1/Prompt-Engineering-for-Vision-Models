import time
import torch
from ultralytics import YOLO

#img = "https://ultralytics.com/images/bus.jpg" # Changed to clean up output
# bash: wget https://ultralytics.com/images/bus.jpg
img = "bus.jpg"


import os
import cv2

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