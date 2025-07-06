# Building Red Light Detection Executable

## Platform-Specific Instructions

### Windows (.exe)
To create a Windows executable:

1. **Install Python 3.8+** on Windows machine
2. **Clone the repository:**
   ```bash
   git clone https://github.com/Bit-Spectre01/red-light-detection.git
   cd red-light-detection
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install pyinstaller
   ```
4. **Run build script:**
   ```bash
   python build_exe.py
   ```
5. **Executable location:** `dist/RedLightDetection/RedLightDetection.exe`

### Linux
The current build creates a Linux executable:

1. **Run build script:**
   ```bash
   python build_exe.py
   ```
2. **Executable location:** `dist/RedLightDetection/RedLightDetection`

## Build Output
- **Executable size:** ~50MB
- **Total distribution:** ~5-6GB (includes PyTorch, OpenCV, YOLO model)
- **Build time:** 3-5 minutes depending on system

## Testing the Executable
```bash
# Show help
./RedLightDetection --help

# Test with video file
./RedLightDetection --video sample.mp4

# Setup ROIs only
./RedLightDetection --setup-roi --video sample.mp4
```

## Distribution
- Copy the entire `dist/` folder to target machines
- No Python installation required on target machines
- All dependencies are bundled in the executable

## Troubleshooting
- **Build fails:** Ensure all dependencies are installed
- **Large size:** Normal due to ML dependencies (PyTorch, OpenCV)
- **Slow build:** PyInstaller analyzes many dependencies - be patient
- **Missing YOLO model:** Run the Python version once to download yolov5nu.pt
