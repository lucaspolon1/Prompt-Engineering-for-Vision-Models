import time
import torch
from ultralytics import YOLO
import argparse
from benchmark_helpers import run_yolo, get_model_accuracy

# personal bash: salloc --gres=gpu:1 --mem=16G --time=00:10:00, nvidia-smi

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

