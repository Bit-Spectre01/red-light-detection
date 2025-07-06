"""
Headless test script for Red Light Violation Detection System

This script tests core functionality without GUI components.
"""

import os
import json
import datetime
from logger import ViolationLogger


def test_logger_headless():
    """Test the violation logger without GUI."""
    print("Testing violation logger...")
    
    logger = ViolationLogger("test_output", "test_video", fps=30)
    
    screenshot_path = logger.get_screenshot_path(1, 123, "Straight Through")
    print(f"Generated screenshot path: {screenshot_path}")
    
    logger.log_violation(
        lane=1,
        vehicle_id=123,
        movement_type="Straight Through",
        violation_type="Red Light Violation",
        screenshot_path=screenshot_path,
        video_path="test_output/test_video/Straight Through/Lane1Vehicle123_20250706_045033_violation.mp4"
    )
    
    logger.print_summary()
    
    print("Logger test completed successfully")
    return True


def test_roi_config():
    """Test ROI configuration loading."""
    print("Testing ROI configuration...")
    
    roi_config = {
        "1": {
            "lane": [(50, 200), (300, 200), (300, 480), (50, 480)],
            "stop_bar": [(50, 400), (300, 400), (300, 410), (50, 410)],
            "red_light": [(90, 40), (110, 40), (110, 60), (90, 60)],
            "green_light": [(310, 40), (330, 40), (330, 60), (310, 60)]
        }
    }
    
    os.makedirs("test_config", exist_ok=True)
    config_path = "test_config/test_roi.json"
    with open(config_path, 'w') as f:
        json.dump(roi_config, f, indent=2)
    
    with open(config_path, 'r') as f:
        loaded_config = json.load(f)
    
    print(f"ROI config saved and loaded successfully: {len(loaded_config)} lanes")
    return True


def test_main_cli():
    """Test main CLI help functionality."""
    print("Testing main CLI...")
    
    import subprocess
    result = subprocess.run(['python', 'main.py', '--help'], 
                          capture_output=True, text=True)
    
    if result.returncode == 0 and "Red Light Violation Detection System" in result.stdout:
        print("CLI help test passed")
        return True
    else:
        print(f"CLI help test failed: {result.stderr}")
        return False


def run_headless_tests():
    """Run all headless tests."""
    print("=== Headless Red Light Detection System Tests ===\n")
    
    tests = [
        ("Logger", test_logger_headless),
        ("ROI Config", test_roi_config),
        ("Main CLI", test_main_cli),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            if test_func():
                print(f"✓ {test_name} test PASSED\n")
                passed += 1
            else:
                print(f"✗ {test_name} test FAILED\n")
        except Exception as e:
            print(f"✗ {test_name} test FAILED with exception: {e}\n")
    
    print(f"=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("All headless tests passed! Core system is ready.")
    else:
        print("Some tests failed. Please check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    run_headless_tests()
