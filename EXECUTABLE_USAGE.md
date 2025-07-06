# Red Light Detection System - Executable Usage

## Overview
This document explains how to use the standalone executable version of the Red Light Detection System.

## System Requirements
- Windows 10/11 (64-bit)
- Minimum 4GB RAM (8GB recommended for large videos)
- 1GB free disk space for temporary files and output

## Installation
1. Download the `dist` folder containing `RedLightDetection.exe`
2. No additional installation required - all dependencies are bundled

## Usage

### Basic Commands
```bash
# Show help and available options
RedLightDetection.exe --help

# Process a single video file
RedLightDetection.exe --video path/to/video.mp4

# Download and process YouTube video
RedLightDetection.exe --youtube "https://www.youtube.com/watch?v=VIDEO_ID"

# Process multiple videos in a directory
RedLightDetection.exe --batch --input-dir path/to/videos/

# Setup ROIs only (without processing)
RedLightDetection.exe --video path/to/video.mp4 --setup-roi
```

### Advanced Options
```bash
# Process without live preview (faster)
RedLightDetection.exe --video path/to/video.mp4 --no-preview

# Save annotated video with detections
RedLightDetection.exe --video path/to/video.mp4 --save-annotated

# Specify custom output directory
RedLightDetection.exe --video path/to/video.mp4 --output-dir custom_output/
```

## Output
- **Violations CSV**: Detailed log of all detected violations
- **Screenshots**: Evidence images for each violation
- **10-second videos**: Video clips showing 5 seconds before/after each violation
- **Annotated videos**: Full video with detection overlays (if --save-annotated used)

## Troubleshooting
- If the executable doesn't start, ensure you have Windows 10/11 64-bit
- For large videos, ensure sufficient disk space for temporary files
- If ROI drawing doesn't work, the system may be running in headless mode

## Features Included
- Multi-lane red light violation detection
- Adaptive traffic light monitoring for 12-hour videos (6 AM to 6 PM)
- Vehicle tracking and movement classification
- Right-turn-on-red violation detection with 3-second stop requirement
- 10-second violation video evidence (5 seconds before/after)
- CSV logging with detailed violation metadata
- Screenshot evidence for each violation
- Interactive ROI drawing tool for setup
- YouTube video download and processing
- Batch processing for multiple videos

## Technical Details
- Uses YOLOv5n model for CPU-optimized vehicle detection
- Adaptive lighting compensation for changing conditions
- Temporal smoothing for stable traffic light detection
- Supports up to 7 lanes with individual configuration
- Frame-by-frame processing with violation evidence collection
