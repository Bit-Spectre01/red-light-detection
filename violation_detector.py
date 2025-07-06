"""
Vehicle Detection and Violation Detection System

This module handles:
- Vehicle detection using YOLOv5n
- Vehicle tracking across frames
- Lane assignment using ROI polygons
- Traffic light state monitoring
- Violation detection logic including right-turn-on-red
- Movement classification (straight, left turn, right turn)
"""

import cv2
import numpy as np
import json
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional, Any
import datetime
from collections import defaultdict, deque


class Vehicle:
    def __init__(self, vehicle_id: int, bbox: Tuple[int, int, int, int], lane: int):
        """
        Initialize a vehicle object.
        
        Args:
            vehicle_id: Unique identifier for the vehicle
            bbox: Bounding box (x1, y1, x2, y2)
            lane: Lane number the vehicle is in
        """
        self.id = vehicle_id
        self.bbox = bbox
        self.lane = lane
        self.centroid_history = []
        self.stop_bar_crossed = False
        self.stop_start_frame = None
        self.movement_type = "Unknown"
        self.violation_logged = False
        
    def update_position(self, bbox: Tuple[int, int, int, int], frame_num: int):
        """Update vehicle position and centroid history."""
        self.bbox = bbox
        centroid = self.get_centroid()
        self.centroid_history.append((centroid, frame_num))
        
        if len(self.centroid_history) > 150:
            self.centroid_history.pop(0)
            
    def get_centroid(self) -> Tuple[int, int]:
        """Get centroid of vehicle bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
        
    def classify_movement(self, entry_point: Tuple[int, int], exit_point: Tuple[int, int]) -> str:
        """
        Classify vehicle movement based on entry and exit points.
        
        Args:
            entry_point: Point where vehicle entered the intersection
            exit_point: Point where vehicle exited the intersection
            
        Returns:
            Movement type: "Straight Through", "Left Turn", or "Right Turn"
        """
        dx = exit_point[0] - entry_point[0]
        dy = exit_point[1] - entry_point[1]
        
        if abs(dx) < 50 and abs(dy) < 50:
            return "Straight Through"
        
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        if -45 <= angle <= 45:
            return "Straight Through"
        elif 45 < angle <= 135:
            return "Left Turn"
        elif -135 <= angle < -45:
            return "Right Turn"
        else:
            return "Straight Through"


class TrafficLightMonitor:
    def __init__(self, roi_config: Dict):
        """
        Initialize traffic light monitoring with adaptive lighting support.
        
        Args:
            roi_config: ROI configuration containing light positions
        """
        self.roi_config = roi_config
        self.light_states = {}  # lane -> state history
        self.state_history_length = 15  # Frames to smooth over (increased for stability)
        self.brightness_history = {}  # lane -> brightness values for adaptation
        self.adaptive_params = {}  # lane -> adaptive detection parameters
        
        for lane_num in roi_config.keys():
            lane_key = int(lane_num)
            self.light_states[lane_key] = deque(maxlen=self.state_history_length)
            self.brightness_history[lane_key] = deque(maxlen=30)  # 30 frames for brightness tracking
            self.adaptive_params[lane_key] = {
                'brightness_factor': 1.0,
                'red_hue_tolerance': 5,  # Additional tolerance for red hue
                'green_hue_tolerance': 10,  # Additional tolerance for green hue
                'min_saturation': 50,
                'min_value': 50
            }
            
    def detect_light_state(self, frame: np.ndarray, lane: int) -> str:
        """
        Detect traffic light state for a specific lane with adaptive lighting support.
        
        Args:
            frame: Current video frame
            lane: Lane number
            
        Returns:
            Light state: "red", "green", or "unknown"
        """
        if lane not in self.roi_config:
            return "unknown"
            
        lane_config = self.roi_config[lane]
        
        red_roi = lane_config.get("red_light", [])
        green_roi = lane_config.get("green_light", [])
        
        self._update_adaptive_parameters(frame, lane, red_roi, green_roi)
        
        red_intensity = 0
        green_intensity = 0
        
        if red_roi:
            red_intensity = self._get_adaptive_color_intensity(frame, red_roi, "red", lane)
            
        if green_roi:
            green_intensity = self._get_adaptive_color_intensity(frame, green_roi, "green", lane)
            
        base_threshold = 30  # Lowered base threshold for better sensitivity
        adaptive_threshold = base_threshold * max(0.5, self.adaptive_params[lane]['brightness_factor'])
        
        min_absolute_threshold = 20
        
        if red_intensity > green_intensity and red_intensity > max(adaptive_threshold, min_absolute_threshold):
            state = "red"
        elif green_intensity > red_intensity and green_intensity > max(adaptive_threshold, min_absolute_threshold):
            state = "green"
        else:
            state = "unknown"
            
        self.light_states[lane].append(state)
        return self._get_smoothed_state(lane)
        
    def _update_adaptive_parameters(self, frame: np.ndarray, lane: int, red_roi: List, green_roi: List):
        """
        Update adaptive parameters based on current lighting conditions.
        
        Args:
            frame: Current video frame
            lane: Lane number
            red_roi: Red light ROI points
            green_roi: Green light ROI points
        """
        brightness_values = []
        
        for roi in [red_roi, green_roi]:
            if roi and len(roi) >= 3:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                pts = np.array(roi, np.int32)
                cv2.fillPoly(mask, [pts], 255)
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi_pixels = gray[mask > 0]
                if len(roi_pixels) > 0:
                    brightness_values.append(np.mean(roi_pixels))
        
        if brightness_values:
            avg_brightness = np.mean(brightness_values)
            self.brightness_history[lane].append(avg_brightness)
            
            if len(self.brightness_history[lane]) >= 10:
                recent_brightness = np.mean(list(self.brightness_history[lane])[-10:])
                
                self.adaptive_params[lane]['brightness_factor'] = max(0.3, min(2.5, recent_brightness / 128.0))
                
                if recent_brightness < 60:  # Very low light (dawn/dusk)
                    self.adaptive_params[lane]['min_saturation'] = 25
                    self.adaptive_params[lane]['min_value'] = 25
                    self.adaptive_params[lane]['red_hue_tolerance'] = 8
                    self.adaptive_params[lane]['green_hue_tolerance'] = 15
                elif recent_brightness < 100:  # Low light
                    self.adaptive_params[lane]['min_saturation'] = 35
                    self.adaptive_params[lane]['min_value'] = 35
                    self.adaptive_params[lane]['red_hue_tolerance'] = 6
                    self.adaptive_params[lane]['green_hue_tolerance'] = 12
                elif recent_brightness > 200:  # Very bright (midday sun)
                    self.adaptive_params[lane]['min_saturation'] = 40
                    self.adaptive_params[lane]['min_value'] = 40
                    self.adaptive_params[lane]['red_hue_tolerance'] = 8
                    self.adaptive_params[lane]['green_hue_tolerance'] = 10
                else:  # Normal lighting
                    self.adaptive_params[lane]['min_saturation'] = 30
                    self.adaptive_params[lane]['min_value'] = 30
                    self.adaptive_params[lane]['red_hue_tolerance'] = 8
                    self.adaptive_params[lane]['green_hue_tolerance'] = 12

    def _get_adaptive_color_intensity(self, frame: np.ndarray, roi: List[Tuple[int, int]], color: str, lane: int) -> float:
        """
        Get color intensity in ROI with adaptive color detection for changing lighting.
        
        Args:
            frame: Video frame
            roi: Region of interest polygon points
            color: "red" or "green"
            lane: Lane number for adaptive parameters
            
        Returns:
            Color intensity value
        """
        if len(roi) < 3:
            return 0
            
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = np.array(roi, np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        params = self.adaptive_params[lane]
        min_sat = params['min_saturation']
        min_val = params['min_value']
        
        if color == "red":
            red_tolerance = params['red_hue_tolerance']
            lower_red1 = np.array([0, max(20, min_sat - 10), max(20, min_val - 10)])
            upper_red1 = np.array([min(180, 15 + red_tolerance), 255, 255])
            lower_red2 = np.array([max(0, 165 - red_tolerance), max(20, min_sat - 10), max(20, min_val - 10)])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            color_mask = cv2.bitwise_or(mask1, mask2)
        else:  # green
            green_tolerance = params['green_hue_tolerance']
            lower_green = np.array([max(0, 35 - green_tolerance), max(20, min_sat - 10), max(20, min_val - 10)])
            upper_green = np.array([min(180, 85 + green_tolerance), 255, 255])
            color_mask = cv2.inRange(hsv, lower_green, upper_green)
            
        combined_mask = cv2.bitwise_and(mask, color_mask)
        
        intensity = np.sum(combined_mask) / 255.0
        
        compensated_intensity = intensity * params['brightness_factor']
        
        return min(compensated_intensity, 1000.0)  # Cap at reasonable maximum

    def _get_color_intensity(self, frame: np.ndarray, roi: List[Tuple[int, int]], color: str) -> float:
        """
        Legacy method for backward compatibility - uses adaptive detection with default lane.
        """
        default_lane = list(self.roi_config.keys())[0] if self.roi_config else 1
        return self._get_adaptive_color_intensity(frame, roi, color, default_lane)
        
    def _get_smoothed_state(self, lane: int) -> str:
        """Get smoothed traffic light state using enhanced temporal filtering for stability."""
        if lane not in self.light_states or len(self.light_states[lane]) == 0:
            return "unknown"
            
        states = list(self.light_states[lane])
        
        if len(states) >= 8:  # Need sufficient history for reliable smoothing
            red_count = states.count("red")
            green_count = states.count("green")
            unknown_count = states.count("unknown")
            
            total_states = len(states)
            
            # Require stronger consensus (60%) for definitive state detection
            red_ratio = red_count / total_states
            green_ratio = green_count / total_states
            
            if red_ratio >= 0.6:
                return "red"
            elif green_ratio >= 0.6:
                return "green"
            elif red_ratio >= 0.4 and red_ratio > green_ratio:
                return "red"  # Lean towards red for safety
            elif green_ratio >= 0.4 and green_ratio > red_ratio:
                return "green"
            else:
                return "unknown"  # Conservative approach during uncertain periods
        else:
            states = list(self.light_states[lane])
            red_count = states.count("red")
            green_count = states.count("green")
            
            if red_count > green_count:
                return "red"
            elif green_count > red_count:
                return "green"
            else:
                return "unknown"


class ViolationDetector:
    def __init__(self, roi_config_path: str, output_dir: str, video_name: str):
        """
        Initialize violation detection system.
        
        Args:
            roi_config_path: Path to ROI configuration JSON file
            output_dir: Output directory for violations
            video_name: Name of video being processed
        """
        with open(roi_config_path, 'r') as f:
            self.roi_config = json.load(f)
            
        self.roi_config = {int(k): v for k, v in self.roi_config.items()}
        
        self.output_dir = output_dir
        self.video_name = video_name
        
        self.model = YOLO('yolov5nu.pt')  # YOLOv5n ultralytics format
        
        self.light_monitor = TrafficLightMonitor(self.roi_config)
        
        self.vehicles = {}  # vehicle_id -> Vehicle object
        self.next_vehicle_id = 1
        self.frame_count = 0
        
        self.stop_duration_frames = 90  # 3 seconds at 30 FPS
        
    def point_in_polygon(self, point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        """Check if point is inside polygon using ray casting algorithm."""
        if len(polygon) < 3:
            return False
            
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
        
    def get_vehicle_lane(self, centroid: Tuple[int, int]) -> Optional[int]:
        """Determine which lane a vehicle is in based on its centroid."""
        for lane_num, lane_config in self.roi_config.items():
            lane_polygon = lane_config.get("lane", [])
            if lane_polygon and self.point_in_polygon(centroid, lane_polygon):
                return lane_num
        return None
        
    def detect_vehicles(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect vehicles in frame using YOLO.
        
        Args:
            frame: Video frame
            
        Returns:
            List of detections (x1, y1, x2, y2, confidence)
        """
        results = self.model(frame, classes=[2, 5, 7])  # car, bus, truck
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    if conf > 0.5:  # Confidence threshold
                        detections.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
                        
        return detections
        
    def update_vehicles(self, detections: List[Tuple[int, int, int, int, float]]):
        """Update vehicle tracking with new detections."""
        unmatched_detections = detections.copy()
        matched_vehicles = set()
        
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle_id in matched_vehicles:
                continue
                
            best_match = None
            best_distance = float('inf')
            
            vehicle_centroid = vehicle.get_centroid()
            
            for i, detection in enumerate(unmatched_detections):
                x1, y1, x2, y2, conf = detection
                det_centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                distance = np.sqrt((vehicle_centroid[0] - det_centroid[0])**2 + 
                                 (vehicle_centroid[1] - det_centroid[1])**2)
                
                if distance < best_distance and distance < 100:  # Max distance threshold
                    best_distance = distance
                    best_match = i
                    
            if best_match is not None:
                detection = unmatched_detections.pop(best_match)
                x1, y1, x2, y2, conf = detection
                vehicle.update_position((x1, y1, x2, y2), self.frame_count)
                matched_vehicles.add(vehicle_id)
                
        for detection in unmatched_detections:
            x1, y1, x2, y2, conf = detection
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            lane = self.get_vehicle_lane(centroid)
            
            if lane is not None:
                vehicle = Vehicle(self.next_vehicle_id, (x1, y1, x2, y2), lane)
                vehicle.update_position((x1, y1, x2, y2), self.frame_count)
                self.vehicles[self.next_vehicle_id] = vehicle
                self.next_vehicle_id += 1
                
        vehicles_to_remove = []
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle_id not in matched_vehicles:
                if len(vehicle.centroid_history) > 0:
                    last_frame = vehicle.centroid_history[-1][1]
                    if self.frame_count - last_frame > 30:
                        vehicles_to_remove.append(vehicle_id)
                        
        for vehicle_id in vehicles_to_remove:
            del self.vehicles[vehicle_id]
            
    def check_stop_bar_crossing(self, vehicle: Vehicle, lane: int, light_state: str) -> bool:
        """
        Check if vehicle crossed stop bar during red light.
        
        Args:
            vehicle: Vehicle object
            lane: Lane number
            light_state: Current traffic light state
            
        Returns:
            True if violation detected
        """
        if lane not in self.roi_config:
            return False
            
        stop_bar = self.roi_config[lane].get("stop_bar", [])
        if not stop_bar:
            return False
            
        centroid = vehicle.get_centroid()
        
        if len(stop_bar) >= 2:
            stop_y = sum(point[1] for point in stop_bar) / len(stop_bar)
            
            if not vehicle.stop_bar_crossed and centroid[1] < stop_y:
                vehicle.stop_bar_crossed = True
                
                if light_state == "red":
                    return True
                    
        return False
        
    def check_right_turn_violation(self, vehicle: Vehicle, lane: int, light_state: str) -> bool:
        """
        Check for right turn on red without proper stop.
        
        Args:
            vehicle: Vehicle object
            lane: Lane number  
            light_state: Current traffic light state
            
        Returns:
            True if violation detected
        """
        if light_state != "red":
            return False
            
        if len(vehicle.centroid_history) < 10:
            return False
            
        centroid = vehicle.get_centroid()
        
        if lane in self.roi_config:
            stop_bar = self.roi_config[lane].get("stop_bar", [])
            if stop_bar:
                stop_y = sum(point[1] for point in stop_bar) / len(stop_bar)
                
                if abs(centroid[1] - stop_y) < 50:
                    if vehicle.stop_start_frame is None:
                        vehicle.stop_start_frame = self.frame_count
                    else:
                        stop_duration = self.frame_count - vehicle.stop_start_frame
                        if stop_duration >= self.stop_duration_frames:
                            return False  # Proper stop, no violation
                else:
                    if vehicle.stop_start_frame is not None:
                        stop_duration = self.frame_count - vehicle.stop_start_frame
                        if stop_duration < self.stop_duration_frames:
                            vehicle.movement_type = "Right Turn"
                            return True  # Violation: didn't stop long enough
                        vehicle.stop_start_frame = None
                        
        return False
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame for violation detection.
        
        Args:
            frame: Video frame
            
        Returns:
            Tuple of (annotated_frame, violations_list)
        """
        self.frame_count += 1
        violations = []
        
        detections = self.detect_vehicles(frame)
        
        self.update_vehicles(detections)
        
        light_states = {}
        for lane in self.roi_config.keys():
            light_states[lane] = self.light_monitor.detect_light_state(frame, lane)
            
        for vehicle_id, vehicle in self.vehicles.items():
            lane = vehicle.lane
            light_state = light_states.get(lane, "unknown")
            
            if self.check_stop_bar_crossing(vehicle, lane, light_state):
                if not vehicle.violation_logged:
                    violations.append({
                        'vehicle_id': vehicle_id,
                        'lane': lane,
                        'violation_type': 'Red Light Violation',
                        'movement_type': 'Straight Through',
                        'frame': self.frame_count,
                        'bbox': vehicle.bbox
                    })
                    vehicle.violation_logged = True
                    
            if self.check_right_turn_violation(vehicle, lane, light_state):
                if not vehicle.violation_logged:
                    violations.append({
                        'vehicle_id': vehicle_id,
                        'lane': lane,
                        'violation_type': 'Right Turn on Red Without Stop',
                        'movement_type': 'Right Turn',
                        'frame': self.frame_count,
                        'bbox': vehicle.bbox
                    })
                    vehicle.violation_logged = True
                    
        annotated_frame = self.annotate_frame(frame, light_states)
        
        return annotated_frame, violations
        
    def annotate_frame(self, frame: np.ndarray, light_states: Dict[int, str]) -> np.ndarray:
        """Annotate frame with vehicle bounding boxes and ROIs."""
        annotated = frame.copy()
        
        for lane, lane_config in self.roi_config.items():
            for roi_type, polygon in lane_config.items():
                if polygon:
                    pts = np.array(polygon, np.int32)
                    color = (0, 255, 0) if roi_type == "lane" else (255, 0, 0)
                    cv2.polylines(annotated, [pts], True, color, 2)
                    
        for vehicle_id, vehicle in self.vehicles.items():
            x1, y1, x2, y2 = vehicle.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated, f"ID:{vehicle_id} L:{vehicle.lane}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                       
        y_offset = 30
        for lane, state in light_states.items():
            color = (0, 0, 255) if state == "red" else (0, 255, 0) if state == "green" else (128, 128, 128)
            cv2.putText(annotated, f"Lane {lane}: {state}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
            
        return annotated
