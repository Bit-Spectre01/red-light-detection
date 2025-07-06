"""
Build script for creating executable from Red Light Detection System

This script creates a PyInstaller spec file and builds the executable
with all necessary dependencies and data files.
"""

import os
import sys
import subprocess
from pathlib import Path

def create_spec_file():
    """Create PyInstaller spec file with all necessary configurations."""
    
    spec_content = '''

block_cipher = None

hidden_imports = [
    'ultralytics',
    'ultralytics.models',
    'ultralytics.models.yolo',
    'ultralytics.models.yolo.detect',
    'ultralytics.utils',
    'ultralytics.utils.downloads',
    'torch',
    'torchvision',
    'cv2',
    'numpy',
    'pandas',
    'PIL',
    'yaml',
    'requests',
    'tqdm',
    'matplotlib',
    'seaborn',
    'scipy',
    'sklearn',
    'yt_dlp',
    'json',
    'datetime',
    'collections',
    'typing',
    'argparse',
    'os',
    'sys'
]

datas = [
    ('yolov5nu.pt', '.'),  # YOLO model file
    ('README.md', '.'),
    ('requirements.txt', '.'),
]

binaries = []

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='RedLightDetection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''
    
    with open('RedLightDetection.spec', 'w') as f:
        f.write(spec_content)
    
    print("✓ Created PyInstaller spec file: RedLightDetection.spec")

def build_executable():
    """Build the executable using PyInstaller."""
    
    print("Building executable with PyInstaller...")
    print("This may take several minutes...")
    
    try:
        result = subprocess.run([
            'pyinstaller', 
            '--clean',
            'RedLightDetection.spec'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Executable built successfully!")
            
            exe_path = Path('dist/RedLightDetection.exe')
            if exe_path.exists():
                size_mb = exe_path.stat().st_size / (1024 * 1024)
                print(f"✓ Executable created: {exe_path}")
                print(f"✓ Size: {size_mb:.1f} MB")
                return True
            else:
                print("✗ Executable file not found in dist/ directory")
                return False
        else:
            print("✗ Build failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Error during build: {e}")
        return False

def main():
    """Main build process."""
    print("=== Red Light Detection System - Executable Builder ===")
    print()
    
    if not os.path.exists('main.py'):
        print("✗ Error: main.py not found. Please run this script from the project directory.")
        return 1
    
    if not os.path.exists('yolov5nu.pt'):
        print("✗ Error: YOLO model file (yolov5nu.pt) not found.")
        print("Please run the system once to download the model:")
        print("python main.py --help")
        return 1
    
    print("✓ Project files found")
    print("✓ YOLO model file found")
    print()
    
    create_spec_file()
    print()
    
    success = build_executable()
    print()
    
    if success:
        print("=== BUILD SUCCESSFUL ===")
        print()
        print("Your executable is ready:")
        if os.name == 'nt':  # Windows
            print("📁 Location: dist/RedLightDetection/RedLightDetection.exe")
            exe_name = "RedLightDetection.exe"
        else:  # Linux/Unix
            print("📁 Location: dist/RedLightDetection/RedLightDetection")
            exe_name = "RedLightDetection"
        print()
        print("Usage:")
        print("1. Copy the entire 'dist' folder to your target machine")
        print(f"2. Run: {exe_name} --help")
        print(f"3. Example: {exe_name} --video sample.mp4")
        print()
        print("Note: The executable includes all dependencies and the YOLO model.")
        return 0
    else:
        print("=== BUILD FAILED ===")
        print("Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
