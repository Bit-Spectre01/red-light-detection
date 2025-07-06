"""
Test script for Red Light Violation Detection System

This script provides basic testing functionality to verify system components.
"""

import cv2
import numpy as np
import os
import json
from roi_tool import ROIDrawingTool
from logger import ViolationLogger
from violation_detector import ViolationDetector


def create_test_frame():
    """Create a test frame for ROI drawing."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    cv2.line(frame, (100, 400), (540, 400), (255, 255, 255), 2)  # Stop line
    cv2.line(frame, (50, 0), (50, 480), (255, 255, 255), 2)      # Lane divider
    cv2.line(frame, (320, 0), (320, 480), (255, 255, 255), 2)    # Center line
    cv2.line(frame, (590, 0), (590, 480), (255, 255, 255), 2)    # Right edge
    
    cv2.circle(frame, (100, 50), 10, (0, 0, 255), -1)    # Red light
    cv2.circle(frame, (320, 50), 10, (0, 255, 0), -1)    # Green light
    cv2.circle(frame, (540, 50), 10, (0, 0, 255), -1)    # Red light
    
    cv2.putText(frame, "Test Intersection", (200, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame


def test_roi_tool():
    """Test the ROI drawing tool with a synthetic frame."""
    print("Testing ROI drawing tool...")
    
    frame = create_test_frame()
    roi_tool = ROIDrawingTool(frame, "test_video")
    
    print("ROI tool initialized successfully")
    print("Note: In actual usage, you would call roi_tool.run() to start interactive drawing")
    
    return True


def test_logger():
    """Test the violation logger."""
    print("Testing violation logger...")
    
    logger = ViolationLogger("test_output", "test_video")
    
    screenshot_path = logger.get_screenshot_path(1, 123, "Straight Through")
    logger.log_violation(
        lane=1,
        vehicle_id=123,
        movement_type="Straight Through",
        violation_type="Red Light Violation",
        screenshot_path=screenshot_path
    )
    
    logger.print_summary()
    
    print("Logger test completed successfully")
    return True


def test_violation_detector():
    """Test the violation detector with mock data."""
    print("Testing violation detector...")
    
    roi_config = {
        1: {
            "lane": [(50, 200), (300, 200), (300, 480), (50, 480)],
            "stop_bar": [(50, 400), (300, 400)],
            "red_light": [(90, 40), (110, 40), (110, 60), (90, 60)],
            "green_light": [(310, 40), (330, 40), (330, 60), (310, 60)]
        }
    }
    
    os.makedirs("test_config", exist_ok=True)
    config_path = "test_config/test_roi.json"
    with open(config_path, 'w') as f:
        json.dump(roi_config, f)
    
    try:
        detector = ViolationDetector(config_path, "test_output", "test_video")
        print("Violation detector initialized successfully")
        
        frame = create_test_frame()
        annotated_frame, violations = detector.process_frame(frame)
        
        print(f"Processed frame, found {len(violations)} violations")
        print("Violation detector test completed successfully")
        
        return True
        
    except Exception as e:
        print(f"Error testing violation detector: {e}")
        return False


def test_yolo_model():
    """Test YOLO model loading."""
    print("Testing YOLO model...")
    
    try:
        from ultralytics import YOLO
        model = YOLO('yolov5nu.pt')
        print("YOLO model loaded successfully")
        
        frame = create_test_frame()
        results = model(frame)
        print(f"YOLO inference completed, found {len(results)} result sets")
        
        return True
        
    except Exception as e:
        print(f"Error testing YOLO model: {e}")
        return False


def run_all_tests():
    """Run all system tests."""
    print("=== Red Light Detection System Tests ===\n")
    
    tests = [
        ("ROI Tool", test_roi_tool),
        ("Logger", test_logger),
        ("YOLO Model", test_yolo_model),
        ("Violation Detector", test_violation_detector),
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
        print("All tests passed! System is ready for use.")
    else:
        print("Some tests failed. Please check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    run_all_tests()
