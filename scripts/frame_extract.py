from typing import Any, Dict, Optional
import os
import subprocess
import cv2
from datetime import timedelta
from smolagents import Tool

class VideoFrameExtractorTool(Tool):
    name = "Video_Frame_Extractor_Tool"
    description = "Download YouTube videos and extract frames for analysis"
    inputs = {
        "video_url": {
            "type": "string",
            "description": "YouTube video URL to process",
            "nullable": True
        },
        "output_dir": {
            "type": "string",
            "description": "Directory to save extracted frames",
            "nullable": True
        },
        "fps_extract": {
            "type": "integer",
            "description": "Number of frames to extract per second",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.OUTPUT_ROOT = "video_frames"
        os.makedirs(self.OUTPUT_ROOT, exist_ok=True)

    def _download_youtube_video(self, url: str, output_filename: str = "video.mp4") -> Dict[str, Any]:
        """Download a YouTube video using yt-dlp."""
        try:
            subprocess.run(["yt-dlp", "-f", "best", url, "-o", output_filename], check=True)
            
            video_info = subprocess.check_output(["yt-dlp", "--print", "duration,title,resolution", url], text=True)
            info_parts = video_info.strip().split()
            duration = float(info_parts[0]) if info_parts[0].replace('.', '', 1).isdigit() else 0
            title = ' '.join(info_parts[1:-1]) if len(info_parts) > 2 else "Unknown"
            resolution = info_parts[-1] if len(info_parts) > 1 else "Unknown"
            
            return {
                "success": True,
                "title": title,
                "duration": duration,
                "resolution": resolution,
                "file_path": os.path.abspath(output_filename)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _extract_video_frames(self, video_path: str, output_dir: str, fps_extract: int) -> Dict[str, Any]:
        """Extract frames from a video at regular intervals."""
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"success": False, "error": "Could not open video file"}
                
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / video_fps
            
            frame_interval = int(video_fps / fps_extract)
            if frame_interval < 1:
                frame_interval = 1
                
            frame_count = 0
            saved_count = 0
            frame_paths = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / video_fps
                    timestamp_str = str(timedelta(seconds=timestamp)).split('.')[0]
                    
                    frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}_{timestamp_str.replace(':', '-')}.jpg")
                    cv2.imwrite(frame_path, frame)
                    
                    frame_paths.append(frame_path)
                    saved_count += 1
                    
                frame_count += 1
                
            cap.release()
            
            return {
                "success": True,
                "total_frames_extracted": saved_count,
                "video_fps": video_fps,
                "video_duration": video_duration,
                "frame_paths": frame_paths,
                "output_directory": os.path.abspath(output_dir)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _create_analysis_summary(self, video_info: Dict[str, Any], frames_info: Dict[str, Any], output_file: str) -> Dict[str, Any]:
        """Create a summary file for the analysis task."""
        try:
            with open(output_file, "w") as f:
                f.write("=== VIDEO ANALYSIS SUMMARY ===\n\n")
                f.write(f"Video Title: {video_info.get('title', 'Unknown')}\n")
                f.write(f"Video URL: {video_info.get('url', 'Unknown')}\n")
                f.write(f"Video Resolution: {video_info.get('resolution', 'Unknown')}\n")
                f.write(f"Video Duration: {frames_info.get('video_duration', 0)} seconds\n")
                f.write(f"Video FPS: {frames_info.get('video_fps', 0)}\n\n")
                f.write(f"Frames Extracted: {frames_info.get('total_frames_extracted', 0)}\n")
                f.write(f"Frames Location: {frames_info.get('output_directory', 'Unknown')}\n\n")
                f.write("Frame Naming Convention:\n
                - Format: frame_XXXX_HH-MM-SS.jpg \n
                - XXXX: 4-digit number, indicating the N-th extracted frame (starting from 0000, incrementing by 1) \n
                - HH-MM-SS: Timestamp in the video (hours-minutes-seconds) when the frame was captured \n
                Example: frame_0002_0-00-02.jpg means the 3nd extracted frame, captured at 2 seconds in the video \n")

                
            return {
                "success": True,
                "summary_path": os.path.abspath(output_file)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def forward(self, video_url: Optional[str] = None, output_dir: Optional[str] = None, fps_extract: Optional[int] = None) -> str:
        """Main processing function"""
        try:
            if video_url is None:
                return "Error: No video URL provided"
            
            # Set default values if None
            output_dir = output_dir or "frames"
            fps_extract = fps_extract or 1
            
            # Download the video
            video_info = self._download_youtube_video(video_url)
            if not video_info.get("success", False):
                return f"Error downloading video: {video_info.get('error', 'Unknown error')}"
            
            # Extract frames from the video
            frames_info = self._extract_video_frames(video_info["file_path"], output_dir, fps_extract)
            if not frames_info.get("success", False):
                return f"Error extracting frames: {frames_info.get('error', 'Unknown error')}"
            
            # Create analysis summary
            video_info["url"] = video_url
            summary_info = self._create_analysis_summary(
                video_info, 
                frames_info,
                os.path.join(self.OUTPUT_ROOT, f"analysis_summary_{os.path.basename(video_info['file_path'])}.txt")
            )
            
            if not summary_info.get("success", False):
                return f"Error creating summary: {summary_info.get('error', 'Unknown error')}"
            
            return f"""Video analysis complete!

Video Title: {video_info.get('title', 'Unknown')}
Resolution: {video_info.get('resolution', 'Unknown')}
Duration: {video_info.get('duration', 0)} seconds

Extracted {frames_info.get('total_frames_extracted', 0)} frames to: {frames_info.get('output_directory', output_dir)}
Summary file created at: {summary_info.get('summary_path', 'Unknown')}

Frame Naming Convention:
- Format: frame_XXXX_HH-MM-SS.jpg
- XXXX: 4-digit number, indicating the N-th extracted frame (starting from 0000, incrementing by 1)
- HH-MM-SS: Timestamp in the video (hours-minutes-seconds) when the frame was captured
Example: frame_0002_0-00-02.jpg means the 3nd extracted frame, captured at 2 seconds in the video

Please review the extracted frames to do the task.
The frames are timestamped in their filenames for easy reference.
"""
                
        except Exception as e:
            return f"Video processing error: {str(e)}"

