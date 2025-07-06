"""
ROI Drawing Tool for Red Light Violation Detection System

This module provides an interactive OpenCV-based interface for drawing polygons
on video frames to define regions of interest (ROIs) including:
- Lane boundaries
- Stop bars  
- Red light positions
- Green light positions

Usage:
- Left click to add points to current polygon
- Press ENTER to confirm current polygon
- Press 'o' to reset current polygon
- Press 'c' to close and save configuration
"""

import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any


class ROIDrawingTool:
    def __init__(self, frame: np.ndarray, video_name: str):
        """
        Initialize ROI drawing tool with a video frame.
        
        Args:
            frame: First frame of video for ROI drawing
            video_name: Name of video file (used for config filename)
        """
        self.frame = frame.copy()
        self.original_frame = frame.copy()
        self.video_name = video_name
        self.current_polygon = []
        self.roi_data = {}
        self.current_lane = 1
        self.current_roi_type = "lane"  # lane, stop_bar, red_light, green_light
        self.roi_types = ["lane", "stop_bar", "red_light", "green_light"]
        self.roi_type_index = 0
        self.colors = {
            "lane": (0, 255, 0),        # Green
            "stop_bar": (255, 0, 0),    # Blue  
            "red_light": (0, 0, 255),   # Red
            "green_light": (0, 255, 255) # Yellow
        }
        self.setup_window()
        
    def setup_window(self):
        """Setup OpenCV window and mouse callback."""
        cv2.namedWindow("ROI Drawing Tool", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("ROI Drawing Tool", self.mouse_callback)
        self.update_display()
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for polygon drawing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_polygon.append((x, y))
            self.update_display()
            
    def update_display(self):
        """Update the display with current polygons and instructions."""
        display_frame = self.original_frame.copy()
        
        for lane_num, lane_data in self.roi_data.items():
            for roi_type, polygon in lane_data.items():
                if polygon:
                    pts = np.array(polygon, np.int32)
                    cv2.polylines(display_frame, [pts], True, self.colors[roi_type], 2)
                    if len(polygon) > 0:
                        cv2.putText(display_frame, f"L{lane_num}_{roi_type}", 
                                  polygon[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                  self.colors[roi_type], 1)
        
        if len(self.current_polygon) > 0:
            for i, point in enumerate(self.current_polygon):
                cv2.circle(display_frame, point, 3, self.colors[self.current_roi_type], -1)
                if i > 0:
                    cv2.line(display_frame, self.current_polygon[i-1], point, 
                           self.colors[self.current_roi_type], 2)
            
            if len(self.current_polygon) > 2:
                cv2.line(display_frame, self.current_polygon[-1], self.current_polygon[0], 
                       self.colors[self.current_roi_type], 1)
        
        instructions = [
            f"Lane {self.current_lane} - Drawing: {self.current_roi_type}",
            "Left click: Add point",
            "ENTER: Confirm polygon", 
            "O: Reset current polygon",
            "C: Close and save",
            "N: Next ROI type",
            "L: Next lane"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(display_frame, instruction, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("ROI Drawing Tool", display_frame)
        
    def next_roi_type(self):
        """Move to next ROI type for current lane."""
        self.roi_type_index = (self.roi_type_index + 1) % len(self.roi_types)
        self.current_roi_type = self.roi_types[self.roi_type_index]
        
    def next_lane(self):
        """Move to next lane."""
        self.current_lane += 1
        self.roi_type_index = 0
        self.current_roi_type = self.roi_types[0]
        
    def confirm_polygon(self):
        """Confirm current polygon and save to ROI data."""
        if len(self.current_polygon) >= 3:
            if self.current_lane not in self.roi_data:
                self.roi_data[self.current_lane] = {
                    "lane": [],
                    "stop_bar": [],
                    "red_light": [],
                    "green_light": []
                }
            
            self.roi_data[self.current_lane][self.current_roi_type] = self.current_polygon.copy()
            self.current_polygon = []
            
            self.next_roi_type()
            
            print(f"Saved {self.current_roi_type} for lane {self.current_lane}")
        else:
            print("Need at least 3 points for a polygon")
            
    def reset_current_polygon(self):
        """Reset the current polygon being drawn."""
        self.current_polygon = []
        print("Reset current polygon")
        
    def save_configuration(self):
        """Save ROI configuration to JSON file."""
        config_dir = "config"
        os.makedirs(config_dir, exist_ok=True)
        
        config_filename = os.path.join(config_dir, f"{self.video_name}_roi.json")
        
        with open(config_filename, 'w') as f:
            json.dump(self.roi_data, f, indent=2)
            
        print(f"ROI configuration saved to {config_filename}")
        return config_filename
        
    def load_configuration(self, config_path: str):
        """Load existing ROI configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                self.roi_data = json.load(f)
            print(f"Loaded ROI configuration from {config_path}")
            self.update_display()
        except FileNotFoundError:
            print(f"No existing configuration found at {config_path}")
        except json.JSONDecodeError:
            print(f"Invalid JSON in configuration file {config_path}")
            
    def run(self):
        """Run the ROI drawing interface."""
        print("ROI Drawing Tool Started")
        print("Instructions:")
        print("- Left click to add points")
        print("- Press ENTER to confirm polygon")
        print("- Press 'o' to reset current polygon") 
        print("- Press 'n' to move to next ROI type")
        print("- Press 'l' to move to next lane")
        print("- Press 'c' to close and save")
        
        config_path = os.path.join("config", f"{self.video_name}_roi.json")
        self.load_configuration(config_path)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # ENTER
                self.confirm_polygon()
                self.update_display()
            elif key == ord('o') or key == ord('O'):
                self.reset_current_polygon()
                self.update_display()
            elif key == ord('n') or key == ord('N'):
                self.next_roi_type()
                self.update_display()
            elif key == ord('l') or key == ord('L'):
                self.next_lane()
                self.update_display()
            elif key == ord('c') or key == ord('C'):
                config_file = self.save_configuration()
                cv2.destroyAllWindows()
                return config_file
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return None
                
        cv2.destroyAllWindows()
        return None


def draw_rois_for_video(video_path: str) -> str:
    """
    Main function to draw ROIs for a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Path to saved configuration file
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Could not read first frame from video: {video_path}")
        
    cap.release()
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    roi_tool = ROIDrawingTool(frame, video_name)
    config_file = roi_tool.run()
    
    return config_file


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python roi_tool.py <video_path>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    config_file = draw_rois_for_video(video_path)
    
    if config_file:
        print(f"ROI configuration saved to: {config_file}")
    else:
        print("ROI drawing cancelled")
