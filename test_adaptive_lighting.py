"""
Test script for adaptive traffic light detection under varying lighting conditions.

This script simulates different lighting scenarios to verify the enhanced
traffic light detection can handle dawn-to-dusk lighting variations.
"""

import cv2
import numpy as np
import json
import os
from violation_detector import TrafficLightMonitor


def create_synthetic_light_frame(brightness_level: int, light_color: str) -> np.ndarray:
    """
    Create a synthetic frame with traffic light at specified brightness.
    
    Args:
        brightness_level: Overall brightness (0-255)
        light_color: "red" or "green"
        
    Returns:
        Synthetic video frame
    """
    frame = np.full((480, 640, 3), brightness_level // 3, dtype=np.uint8)
    
    red_center = (100, 50)
    green_center = (320, 50)
    
    if light_color == "red":
        red_intensity = min(255, brightness_level + 50)
        cv2.circle(frame, red_center, 15, (0, 0, red_intensity), -1)
        cv2.circle(frame, green_center, 15, (0, brightness_level // 4, 0), -1)
    else:  # green
        green_intensity = min(255, brightness_level + 50)
        cv2.circle(frame, green_center, 15, (0, green_intensity, 0), -1)
        cv2.circle(frame, red_center, 15, (brightness_level // 4, 0, 0), -1)
    
    return frame


def test_adaptive_lighting():
    """Test adaptive lighting detection across different brightness levels."""
    print("Testing Adaptive Traffic Light Detection")
    print("=" * 50)
    
    roi_config = {
        1: {
            "lane": [(50, 200), (300, 200), (300, 480), (50, 480)],
            "stop_bar": [(50, 400), (300, 400)],
            "red_light": [(85, 35), (115, 35), (115, 65), (85, 65)],
            "green_light": [(305, 35), (335, 35), (335, 65), (305, 65)]
        }
    }
    
    monitor = TrafficLightMonitor(roi_config)
    
    test_scenarios = [
        (30, "Very Low Light (Dawn)"),
        (60, "Low Light (Early Morning)"),
        (120, "Normal Light"),
        (180, "Bright Light"),
        (220, "Very Bright (Midday Sun)")
    ]
    
    results = []
    
    for brightness, condition in test_scenarios:
        print(f"\nTesting {condition} (Brightness: {brightness})")
        print("-" * 40)
        
        red_detections = []
        for frame_num in range(20):  # Test over multiple frames for stability
            frame = create_synthetic_light_frame(brightness, "red")
            detected_state = monitor.detect_light_state(frame, 1)
            red_detections.append(detected_state)
        
        red_accuracy = red_detections.count("red") / len(red_detections)
        
        green_detections = []
        for frame_num in range(20):
            frame = create_synthetic_light_frame(brightness, "green")
            detected_state = monitor.detect_light_state(frame, 1)
            green_detections.append(detected_state)
        
        green_accuracy = green_detections.count("green") / len(green_detections)
        
        print(f"Red Light Detection Accuracy: {red_accuracy:.2%}")
        print(f"Green Light Detection Accuracy: {green_accuracy:.2%}")
        
        params = monitor.adaptive_params[1]
        print(f"Adaptive Parameters:")
        print(f"  Brightness Factor: {params['brightness_factor']:.2f}")
        print(f"  Min Saturation: {params['min_saturation']}")
        print(f"  Red Hue Tolerance: {params['red_hue_tolerance']}")
        print(f"  Green Hue Tolerance: {params['green_hue_tolerance']}")
        
        results.append({
            'condition': condition,
            'brightness': brightness,
            'red_accuracy': red_accuracy,
            'green_accuracy': green_accuracy,
            'brightness_factor': params['brightness_factor']
        })
    
    print("\n" + "=" * 50)
    print("ADAPTIVE LIGHTING TEST SUMMARY")
    print("=" * 50)
    
    total_red_accuracy = sum(r['red_accuracy'] for r in results) / len(results)
    total_green_accuracy = sum(r['green_accuracy'] for r in results) / len(results)
    
    print(f"Overall Red Light Detection: {total_red_accuracy:.2%}")
    print(f"Overall Green Light Detection: {total_green_accuracy:.2%}")
    
    min_red_accuracy = min(r['red_accuracy'] for r in results)
    min_green_accuracy = min(r['green_accuracy'] for r in results)
    
    print(f"Worst Case Red Detection: {min_red_accuracy:.2%}")
    print(f"Worst Case Green Detection: {min_green_accuracy:.2%}")
    
    success = min_red_accuracy >= 0.8 and min_green_accuracy >= 0.8
    
    if success:
        print("\n✅ ADAPTIVE LIGHTING TEST PASSED")
        print("System successfully handles lighting variations from dawn to dusk!")
    else:
        print("\n❌ ADAPTIVE LIGHTING TEST FAILED")
        print("System needs further tuning for consistent performance.")
    
    return success


def test_lighting_transition():
    """Test detection stability during lighting transitions."""
    print("\n" + "=" * 50)
    print("LIGHTING TRANSITION TEST")
    print("=" * 50)
    
    roi_config = {
        1: {
            "red_light": [(85, 35), (115, 35), (115, 65), (85, 65)],
            "green_light": [(305, 35), (335, 35), (335, 65), (305, 65)]
        }
    }
    
    monitor = TrafficLightMonitor(roi_config)
    
    brightness_progression = list(range(40, 200, 10))  # 40 to 190 in steps of 10
    
    red_states = []
    for brightness in brightness_progression:
        frame = create_synthetic_light_frame(brightness, "red")
        state = monitor.detect_light_state(frame, 1)
        red_states.append(state)
        print(f"Brightness {brightness:3d}: Detected '{state}'")
    
    red_count = red_states.count("red")
    stability = red_count / len(red_states)
    
    print(f"\nTransition Stability: {stability:.2%}")
    
    if stability >= 0.85:
        print("✅ TRANSITION TEST PASSED - Stable detection during lighting changes")
        return True
    else:
        print("❌ TRANSITION TEST FAILED - Unstable detection during transitions")
        return False


if __name__ == "__main__":
    print("Adaptive Traffic Light Detection Test Suite")
    print("Testing enhanced detection for 12-hour video scenarios")
    print("=" * 60)
    
    test1_passed = test_adaptive_lighting()
    test2_passed = test_lighting_transition()
    
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Enhanced traffic light detection is ready for 12-hour videos!")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Further optimization may be needed for challenging lighting conditions.")
    
    print("\nThe system now includes:")
    print("• Adaptive HSV color range adjustment")
    print("• Brightness compensation for dawn-to-dusk variations")
    print("• Enhanced temporal smoothing for stability")
    print("• Wider hue tolerance during low/high light conditions")
