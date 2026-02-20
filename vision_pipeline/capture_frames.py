"""Multi-source video frame acquisition module for PX4 vision pipeline.
Captures time-synchronized image frames from simulator, real UAV camera,
webcam, or video stream (UDP/RTSP/file) at a controlled FPS,
and logs precise timestamps for downstream state estimation.
CSV : frame_id, system_time_sec, image_timestamp_iso, file_name
"""

import cv2
import os
import time
import argparse
import csv
from datetime import datetime
from pathlib import Path

# ==========================================
# Utility Functions
# ==========================================

def create_dirs(base_path: Path):
    frames_dir = base_path / "frames"
    logs_dir = base_path / "logs"
    frames_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return frames_dir, logs_dir


def open_video_source(source: str):
    # Auto-detect integer webcam index
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    return cap


# ==========================================
# Main Capture Class
# ==========================================

class FrameCapture:

    def __init__(self, source, output_dir, target_fps):

        self.source = source
        self.output_dir = Path(output_dir)
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        self.frames_dir, self.logs_dir = create_dirs(self.output_dir)
        self.cap = open_video_source(source)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = self.logs_dir / f"capture_log_{timestamp_str}.csv"

        self.csv_file = open(self.log_file_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "frame_id",
            "system_time_sec",
            "image_timestamp_iso",
            "file_name"
        ])

        self.frame_id = 0
        print("✅ Video source opened")
        print(f"📂 Saving frames to: {self.frames_dir}")
        print(f"📝 Log file: {self.log_file_path}")

    def run(self):

        next_capture_time = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Frame not received. Stopping.")
                break

            current_time = time.time()

            if current_time >= next_capture_time:

                timestamp_iso = datetime.utcnow().isoformat()
                filename = f"frame_{self.frame_id:06d}.png"
                file_path = self.frames_dir / filename

                cv2.imwrite(str(file_path), frame)

                self.csv_writer.writerow([
                    self.frame_id,
                    current_time,
                    timestamp_iso,
                    filename
                ])

                print(f"📸 Saved {filename}")

                self.frame_id += 1
                next_capture_time += self.frame_interval

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        self.csv_file.close()
        print("🛑 Capture stopped and resources released.")


# ==========================================
# CLI Entry
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="Professional Frame Capture for PX4 Vision Pipeline"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: 0 (webcam), udp://..., rtsp://..., or video file path"
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Target capture FPS"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Base output directory (usually vision_pipeline)"
    )

    args = parser.parse_args()

    capture = FrameCapture(
        source=args.source,
        output_dir=args.output,
        target_fps=args.fps
    )

    capture.run()


if __name__ == "__main__":
    main()
capture_frames.py


