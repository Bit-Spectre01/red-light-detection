"""
Logging System for Red Light Violation Detection

This module handles CSV logging of violations with the following columns:
- timestamp: YYYY-MM-DD HH:MM:SS
- lane: Lane number (1-7)
- vehicle_id: Unique vehicle identifier  
- movement_type: "Straight Through", "Left Turn", or "Right Turn"
- violation_type: Type of violation detected
- screenshot_path: Path to evidence screenshot
"""

import csv
import os
import datetime
import cv2
import numpy as np
from typing import Dict, Any, Optional, List


class ViolationLogger:
    def __init__(self, output_dir: str, video_name: str, fps: int = 30):
        """
        Initialize violation logger.
        
        Args:
            output_dir: Base output directory
            video_name: Name of video being processed
            fps: Video frame rate for timing calculations
        """
        self.output_dir = output_dir
        self.video_name = video_name
        self.video_output_dir = os.path.join(output_dir, video_name)
        self.csv_path = os.path.join(self.video_output_dir, "violations.csv")
        self.fps = fps
        self.frame_buffer = []  # Buffer to store recent frames for violation videos
        self.buffer_size = fps * 10  # 10 seconds of frames
        
        os.makedirs(self.video_output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.video_output_dir, "Right Turn"), exist_ok=True)
        os.makedirs(os.path.join(self.video_output_dir, "Straight Through"), exist_ok=True)
        os.makedirs(os.path.join(self.video_output_dir, "Left Turn"), exist_ok=True)
        
        self.initialize_csv()
        
    def initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'timestamp',
                    'lane', 
                    'vehicle_id',
                    'movement_type',
                    'violation_type',
                    'screenshot_path',
                    'video_path'
                ])
                
    def add_frame_to_buffer(self, frame: np.ndarray):
        """
        Add frame to the circular buffer for violation video creation.
        
        Args:
            frame: Video frame to add to buffer
        """
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
    
    def create_violation_video(self, 
                              lane: int,
                              vehicle_id: int,
                              movement_type: str,
                              timestamp: datetime.datetime,
                              violation_frame_index: int) -> str:
        """
        Create a 10-second violation video (5 seconds before/after violation).
        
        Args:
            lane: Lane number
            vehicle_id: Vehicle ID
            movement_type: Movement type for subdirectory
            timestamp: Timestamp for filename
            violation_frame_index: Index of violation frame in buffer
            
        Returns:
            Path to created video file
        """
        if len(self.frame_buffer) < self.fps * 5:  # Need at least 5 seconds of frames
            return ""
            
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"Lane{lane}Vehicle{vehicle_id}_{timestamp_str}_violation.mp4"
        video_path = os.path.join(
            self.video_output_dir,
            movement_type,
            filename
        )
        
        start_frame = max(0, violation_frame_index - self.fps * 5)
        end_frame = min(len(self.frame_buffer), violation_frame_index + self.fps * 5)
        
        if end_frame - start_frame < self.fps * 5:  # Not enough frames
            return ""
            
        height, width = self.frame_buffer[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
        
        try:
            for i in range(start_frame, end_frame):
                if i < len(self.frame_buffer):
                    frame = self.frame_buffer[i].copy()
                    
                    if i == violation_frame_index:
                        cv2.putText(frame, "VIOLATION DETECTED", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        cv2.rectangle(frame, (40, 20), (400, 70), (0, 0, 255), 3)
                    
                    frame_time = timestamp - datetime.timedelta(seconds=(violation_frame_index - i) / self.fps)
                    time_str = frame_time.strftime("%H:%M:%S.%f")[:-3]
                    cv2.putText(frame, time_str, (width - 200, height - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    video_writer.write(frame)
                    
        finally:
            video_writer.release()
            
        return video_path

    def log_violation(self, 
                     lane: int,
                     vehicle_id: int, 
                     movement_type: str,
                     violation_type: str,
                     screenshot_path: str,
                     video_path: str = "",
                     timestamp: Optional[datetime.datetime] = None):
        """
        Log a violation to the CSV file.
        
        Args:
            lane: Lane number where violation occurred
            vehicle_id: Unique identifier for the vehicle
            movement_type: "Straight Through", "Left Turn", or "Right Turn"  
            violation_type: Type of violation (e.g., "Red Light Violation")
            screenshot_path: Path to evidence screenshot
            video_path: Path to violation video clip
            timestamp: Timestamp of violation (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()
            
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                timestamp_str,
                lane,
                vehicle_id, 
                movement_type,
                violation_type,
                screenshot_path,
                video_path
            ])
            
        print(f"Logged violation: Lane {lane}, Vehicle {vehicle_id}, {violation_type}")
        if video_path:
            print(f"  Video evidence: {video_path}")
        
    def get_screenshot_path(self, 
                           lane: int,
                           vehicle_id: int, 
                           movement_type: str,
                           timestamp: Optional[datetime.datetime] = None) -> str:
        """
        Generate screenshot path for a violation.
        
        Args:
            lane: Lane number
            vehicle_id: Vehicle ID
            movement_type: Movement type for subdirectory
            timestamp: Timestamp for filename (defaults to current time)
            
        Returns:
            Full path where screenshot should be saved
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()
            
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        
        filename = f"Lane{lane}Vehicle{vehicle_id}_{timestamp_str}.jpg"
        
        screenshot_path = os.path.join(
            self.video_output_dir,
            movement_type,
            filename
        )
        
        return screenshot_path
        
    def get_violation_count(self) -> int:
        """
        Get total number of violations logged.
        
        Returns:
            Number of violations in CSV file
        """
        try:
            with open(self.csv_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)
                return sum(1 for row in reader)
        except FileNotFoundError:
            return 0
            
    def get_violations_by_lane(self) -> Dict[int, int]:
        """
        Get violation count by lane.
        
        Returns:
            Dictionary mapping lane number to violation count
        """
        lane_counts = {}
        
        try:
            with open(self.csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    lane = int(row['lane'])
                    lane_counts[lane] = lane_counts.get(lane, 0) + 1
        except FileNotFoundError:
            pass
            
        return lane_counts
        
    def get_violations_by_movement(self) -> Dict[str, int]:
        """
        Get violation count by movement type.
        
        Returns:
            Dictionary mapping movement type to violation count
        """
        movement_counts = {}
        
        try:
            with open(self.csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    movement = row['movement_type']
                    movement_counts[movement] = movement_counts.get(movement, 0) + 1
        except FileNotFoundError:
            pass
            
        return movement_counts
        
    def print_summary(self):
        """Print summary of logged violations."""
        total_violations = self.get_violation_count()
        lane_counts = self.get_violations_by_lane()
        movement_counts = self.get_violations_by_movement()
        
        print(f"\n=== Violation Summary for {self.video_name} ===")
        print(f"Total violations: {total_violations}")
        
        if lane_counts:
            print("\nViolations by lane:")
            for lane in sorted(lane_counts.keys()):
                print(f"  Lane {lane}: {lane_counts[lane]}")
                
        if movement_counts:
            print("\nViolations by movement type:")
            for movement, count in movement_counts.items():
                print(f"  {movement}: {count}")
                
        print(f"\nDetailed log: {self.csv_path}")


if __name__ == "__main__":
    logger = ViolationLogger("output", "test_video")
    
    logger.log_violation(
        lane=1,
        vehicle_id=123,
        movement_type="Straight Through", 
        violation_type="Red Light Violation",
        screenshot_path=logger.get_screenshot_path(1, 123, "Straight Through")
    )
    
    logger.log_violation(
        lane=2,
        vehicle_id=456,
        movement_type="Right Turn",
        violation_type="Right Turn on Red Without Stop", 
        screenshot_path=logger.get_screenshot_path(2, 456, "Right Turn")
    )
    
    logger.print_summary()
