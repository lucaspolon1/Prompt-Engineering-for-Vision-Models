import ultralytics
import time
import torch
from ultralytics import YOLO

model_name = "yolov8n.pt" # small & fast
img = "https://ultralytics.com/images/bus.jpg"

def run_yolo(device):
    model = YOLO(model_name)
    model.to(device)

    start = time.time()
    results = model(img)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    return results, end - start

# CPU
cpu_results, cpu_time = run_yolo("cpu")

# GPU
if torch.cuda.is_available():
    gpu_results, gpu_time = run_yolo("cuda")
else:
    gpu_results, gpu_time = None, None

print(f"YOLO CPU time: {cpu_time:.4f} sec")
if gpu_results:
    print(f"YOLO GPU time: {gpu_time:.4f} sec")