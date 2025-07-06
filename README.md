# Multi-Lane Red Light Violation Detection System

A complete computer vision system for detecting red light violations across multiple traffic lanes using YOLOv5 and OpenCV.

## Features

- **ROI Drawing Tool**: Interactive polygon drawing for lane boundaries, stop bars, and traffic lights
- **Vehicle Detection & Tracking**: YOLOv5n-based detection with object tracking across lanes
- **Traffic Light Monitoring**: Color-based red/green light state detection
- **Violation Detection**: Automated detection of red light violations with special right-turn-on-red logic
- **Evidence Collection**: Screenshot capture and CSV logging for all violations
- **Multi-Lane Support**: Supports up to 7 lanes with modular architecture

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The system will automatically download YOLOv5n model on first run.

## Usage

### Basic Usage
```bash
python main.py --video path/to/video.mp4
```

### ROI Setup
On first run with a new video, the system will pause on frame 0 to allow ROI drawing:
- Draw polygons for each lane (lane boundary, stop bar, red light, green light)
- Press ENTER to confirm each polygon
- Press O to reset current polygon
- Press C to close and save configuration

### Batch Processing
```bash
python main.py --batch --input-dir /path/to/videos/
```

## Configuration

ROI configurations are saved as JSON files in the `config/` directory and can be edited manually.

## Output Structure

```
output/
├── {video_name}/
│   ├── Right Turn/
│   ├── Straight Through/
│   ├── Left Turn/
│   └── violations.csv
```

## Violation Detection Logic

- **Stop Bar Crossing**: Detects when vehicle centroid crosses stop bar during red light
- **Right Turn on Red**: Requires 3-second stop (90 frames at 30 FPS) before proceeding
- **Movement Classification**: Automatically classifies as straight, left turn, or right turn

## CSV Log Format

- timestamp: YYYY-MM-DD HH:MM:SS
- lane: Lane number (1-7)
- vehicle_id: Unique vehicle identifier
- movement_type: "Straight Through", "Left Turn", or "Right Turn"
- violation_type: Type of violation detected
- screenshot_path: Path to evidence screenshot

## Troubleshooting

- **CPU Performance**: System optimized for CPU-only processing with YOLOv5n
- **Memory Issues**: Use frame skipping options for long videos
- **ROI Editing**: Edit JSON configuration files to avoid redrawing ROIs
