# YOLO Benchmarking & Object Detection

## Project Overview
Benchmarking YOLO models for speed/accuracy tradeoff analysis, with focus on YOLOv26 for stream monitoring and trash detection.

## File Structure
```
Prompt-Engineering-for-Vision-Models/
└── SICK/         
    ├── README.md
    ├── yolo_benchmark.py    # Main benchmarking script
    ├── benchmark_helpers.py # Helper functions (mAP, plotting, CSV)
    ├── detect.py            # Simple object detection on single image
    ├── bus.jpg              # Test image for benchmarking
    ├── dog.jpg              # Example image for detection
    └── outputs/             # Generated outputs (images, CSVs, plots)
```
## Install Dependencies
```bash
conda create -n yolo python=3.10.19 pip
conda activate yolo
pip install -r requirements.txt
```
## HPC Usage
```bash
# Request GPU allocation
salloc --gres=gpu:1 --mem=16G --time=00:30:00

# Check GPU
nvidia-smi

# Run benchmark
python yolo_benchmark.py gpu
```

## Current Progress

### Completed
- ✅ YOLO speed benchmarking on GPU/CPU
- ✅ mAP accuracy measurement (mAP50, mAP50-95, mAP75)
- ✅ Speed/accuracy tradeoff visualization
- ✅ CSV export of benchmark results
- ✅ Support for YOLOv8, v9, v10, v11, v26 models

### Scripts

**1. yolo_benchmark.py**
- Benchmarks YOLO models for speed and accuracy
- Configurable mAP metric (mAP50, mAP50-95, mAP75)
- Generates comparison plots and CSV results
- Usage: `python yolo_benchmark.py gpu` or `python yolo_benchmark.py cpu`

**2. benchmark_helpers.py**
- `run_yolo()` - Run inference and measure time
- `get_model_accuracy()` - Calculate mAP on COCO validation set
- `save_results_to_csv()` - Export results
- `plot_speed_accuracy_tradeoff()` - Generate visualization

**3. yolo_detect.py**
- Simple object detection on a single image
- Saves annotated output with bounding boxes
- Usage: `python yolo_detect.py dog.jpg`

## Key Configuration
```python
# In yolo_benchmark.py
NUM_WARMUPS = 5          # Warmup iterations before timing
NUM_RUNS = 20            # Number of timed runs
MAP_METRIC = "mAP50"     # Which mAP to optimize for
```

## Next Steps / Open Questions

### TODO
- [ ] Apply object detection to video stream
- [ ] Test YOLOv26 specifically for stream/trash detection
- [ ] Implement video processing pipeline

### Questions for Further Work
- What accuracy threshold is acceptable for real-time stream monitoring?
- How to handle varying lighting conditions in stream footage?
- Best way to track detected objects across frames?

## Notes
- COCO validation dataset (~1GB) downloads on first mAP calculation
- Use `fraction=0.1` in `get_model_accuracy()` for smaller validation subset
- GPU recommended for benchmarking (significantly faster than CPU)