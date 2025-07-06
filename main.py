"""
Main Application for Red Light Violation Detection System

This module provides the command-line interface and orchestrates all components:
- Video loading and processing
- ROI setup workflow
- Violation detection pipeline
- Evidence collection and logging
- Progress reporting
"""

import cv2
import argparse
import os
import sys
from typing import Optional, List
import datetime

from roi_tool import draw_rois_for_video
from violation_detector import ViolationDetector
from logger import ViolationLogger


class RedLightDetectionSystem:
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the red light detection system.
        
        Args:
            output_dir: Base output directory for results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def setup_rois(self, video_path: str) -> Optional[str]:
        """
        Setup ROIs for a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to ROI configuration file or None if cancelled
        """
        print(f"Setting up ROIs for video: {video_path}")
        
        try:
            config_file = draw_rois_for_video(video_path)
            return config_file
        except Exception as e:
            print(f"Error setting up ROIs: {e}")
            return None
            
    def process_video(self, 
                     video_path: str, 
                     roi_config_path: str,
                     show_preview: bool = True,
                     save_annotated: bool = False) -> bool:
        """
        Process a video file for violation detection.
        
        Args:
            video_path: Path to video file
            roi_config_path: Path to ROI configuration file
            show_preview: Whether to show live preview
            save_annotated: Whether to save annotated video
            
        Returns:
            True if processing completed successfully
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        print(f"Processing video: {video_path}")
        print(f"Using ROI config: {roi_config_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return False
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        try:
            detector = ViolationDetector(roi_config_path, self.output_dir, video_name)
            logger = ViolationLogger(self.output_dir, video_name, fps)
        except Exception as e:
            print(f"Error initializing detector: {e}")
            return False
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        annotated_writer = None
        if save_annotated:
            annotated_path = os.path.join(self.output_dir, video_name, f"{video_name}_annotated.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            annotated_writer = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))
            
        frame_count = 0
        violation_count = 0
        
        print("Starting violation detection...")
        print("Press 'q' to quit, 'p' to pause/resume")
        
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame_count += 1
                    
                    logger.add_frame_to_buffer(frame)
                    
                    annotated_frame, violations = detector.process_frame(frame)
                    
                    for violation in violations:
                        violation_count += 1
                        
                        timestamp = datetime.datetime.now() - datetime.timedelta(
                            seconds=(total_frames - frame_count) / fps
                        )
                        
                        screenshot_path = logger.get_screenshot_path(
                            violation['lane'],
                            violation['vehicle_id'],
                            violation['movement_type'],
                            timestamp
                        )
                        
                        cv2.imwrite(screenshot_path, frame)
                        
                        video_path = logger.create_violation_video(
                            violation['lane'],
                            violation['vehicle_id'],
                            violation['movement_type'],
                            timestamp,
                            len(logger.frame_buffer) - 1  # Current frame index in buffer
                        )
                        
                        logger.log_violation(
                            lane=violation['lane'],
                            vehicle_id=violation['vehicle_id'],
                            movement_type=violation['movement_type'],
                            violation_type=violation['violation_type'],
                            screenshot_path=screenshot_path,
                            video_path=video_path,
                            timestamp=timestamp
                        )
                        
                    if annotated_writer is not None:
                        annotated_writer.write(annotated_frame)
                        
                    if show_preview:
                        progress_text = f"Frame: {frame_count}/{total_frames} | Violations: {violation_count}"
                        cv2.putText(annotated_frame, progress_text, (10, height - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        cv2.imshow("Red Light Violation Detection", annotated_frame)
                        
                    if frame_count % 100 == 0:
                        progress = (frame_count / total_frames) * 100
                        print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - Violations: {violation_count}")
                        
                if show_preview:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        paused = not paused
                        print("Paused" if paused else "Resumed")
                else:
                    cv2.waitKey(1)
                    
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
            
        finally:
            cap.release()
            if annotated_writer is not None:
                annotated_writer.release()
            if show_preview:
                cv2.destroyAllWindows()
                
        print(f"\nProcessing completed!")
        print(f"Total frames processed: {frame_count}")
        print(f"Total violations detected: {violation_count}")
        
        logger.print_summary()
        
        return True
        
    def process_batch(self, input_dir: str, show_preview: bool = False) -> bool:
        """
        Process multiple video files in a directory.
        
        Args:
            input_dir: Directory containing video files
            show_preview: Whether to show preview for each video
            
        Returns:
            True if all videos processed successfully
        """
        if not os.path.exists(input_dir):
            print(f"Error: Input directory {input_dir} does not exist")
            return False
            
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []
        
        for file in os.listdir(input_dir):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(input_dir, file))
                
        if not video_files:
            print(f"No video files found in {input_dir}")
            return False
            
        print(f"Found {len(video_files)} video files to process")
        
        success_count = 0
        
        for i, video_path in enumerate(video_files, 1):
            print(f"\n=== Processing video {i}/{len(video_files)}: {os.path.basename(video_path)} ===")
            
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            roi_config_path = os.path.join("config", f"{video_name}_roi.json")
            
            if not os.path.exists(roi_config_path):
                print(f"ROI configuration not found for {video_name}")
                print("Setting up ROIs...")
                roi_config_path = self.setup_rois(video_path)
                
                if roi_config_path is None:
                    print(f"Skipping {video_name} - ROI setup cancelled")
                    continue
                    
            if self.process_video(video_path, roi_config_path, show_preview, save_annotated=True):
                success_count += 1
            else:
                print(f"Failed to process {video_name}")
                
        print(f"\n=== Batch Processing Complete ===")
        print(f"Successfully processed: {success_count}/{len(video_files)} videos")
        
        return success_count == len(video_files)
        
    def download_youtube_video(self, url: str, output_path: str = "videos") -> Optional[str]:
        """
        Download YouTube video for processing.
        
        Args:
            url: YouTube video URL
            output_path: Directory to save downloaded video
            
        Returns:
            Path to downloaded video file or None if failed
        """
        try:
            import yt_dlp
        except ImportError:
            print("yt-dlp not installed. Installing...")
            os.system("pip install yt-dlp")
            import yt_dlp
            
        os.makedirs(output_path, exist_ok=True)
        
        ydl_opts = {
            'format': 'best[height<=720]',  # Download best quality up to 720p
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'video')
                
                print(f"Downloading: {title}")
                ydl.download([url])
                
                for file in os.listdir(output_path):
                    if title.replace('/', '_') in file:
                        return os.path.join(output_path, file)
                        
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None
            
        return None


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Red Light Violation Detection System")
    
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--youtube', type=str, help='YouTube video URL')
    parser.add_argument('--batch', action='store_true', help='Process multiple videos')
    parser.add_argument('--input-dir', type=str, default='videos', 
                       help='Input directory for batch processing')
    
    parser.add_argument('--roi-config', type=str, help='Path to ROI configuration file')
    parser.add_argument('--setup-roi', action='store_true', help='Setup ROIs only')
    parser.add_argument('--no-preview', action='store_true', help='Disable live preview')
    parser.add_argument('--save-annotated', action='store_true', help='Save annotated video')
    
    parser.add_argument('--output-dir', type=str, default='output', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    system = RedLightDetectionSystem(args.output_dir)
    
    if args.youtube:
        print(f"Downloading YouTube video: {args.youtube}")
        video_path = system.download_youtube_video(args.youtube)
        if video_path is None:
            print("Failed to download YouTube video")
            return 1
        args.video = video_path
        
    if args.batch:
        success = system.process_batch(args.input_dir, not args.no_preview)
        return 0 if success else 1
        
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file {args.video} does not exist")
            return 1
            
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        
        if args.roi_config:
            roi_config_path = args.roi_config
        else:
            roi_config_path = os.path.join("config", f"{video_name}_roi.json")
            
        if args.setup_roi or not os.path.exists(roi_config_path):
            print("Setting up ROIs...")
            roi_config_path = system.setup_rois(args.video)
            
            if roi_config_path is None:
                print("ROI setup cancelled")
                return 1
                
            if args.setup_roi:
                print(f"ROI configuration saved to: {roi_config_path}")
                return 0
                
        success = system.process_video(
            args.video, 
            roi_config_path, 
            not args.no_preview,
            args.save_annotated
        )
        
        return 0 if success else 1
        
    print("Error: No input specified. Use --video, --youtube, or --batch")
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
