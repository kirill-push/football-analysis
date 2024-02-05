import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


class VideoFrameData:
    def __init__(self, video_path: str, bbox_paths: Dict[str, str]) -> None:
        self.video_path: str = video_path
        self.bbox_paths: Dict[str, str] = bbox_paths
        self.frames: List[np.ndarray] = []
        self.bboxes: Dict[str, List[List[List[float]]]] = {
            key: [] for key in bbox_paths.keys()
        }
        self.load_video_frames()
        self.load_bboxes()

    def load_video_frames(self) -> None:
        cap: cv2.VideoCapture = cv2.VideoCapture(self.video_path)
        while True:
            ret: bool
            frame: np.ndarray
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
        cap.release()

    def load_bboxes(self) -> None:
        for key, path in self.bbox_paths.items():
            with open(path, "r") as f:
                self.bboxes[key] = json.load(f)
    
    def get_item(self, frame_idx: int) -> Optional[Dict[str, Any]]:
        if frame_idx < 0 or frame_idx >= len(self.frames):
            return None
        frame_data = {"frame": self.frames[frame_idx], "bboxes": {}}
        for key, bboxes_list in self.bboxes.items():
            if frame_idx < len(bboxes_list):
                frame_data["bboxes"][key] = bboxes_list[frame_idx]
        return frame_data

    def visualize_frame(self, frame_idx: int, box_type: str | None = None, bboxes_list: List | None = None) -> None:
        frame_data = self.get_item(frame_idx)
        if frame_data is None:
            print("Frame index out of bounds.")
            return

        frame = frame_data['frame'].copy()
        plt.figure(figsize=(30, 3))
        if bboxes_list is None:
            for key, bboxes in frame_data['bboxes'].items():
                for bbox in bboxes:
                    if box_type is None or key == box_type:
                        x1, y1, x2, y2, _ = bbox
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), self.get_color_by_type(key), 2)
        else:
            if box_type is None:
                box_type = 'pl'
            for bbox in bboxes_list:
                x1, y1, x2, y2, _ = bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), self.get_color_by_type(box_type), 2)

        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    @staticmethod
    def get_color_by_type(object_type: str) -> Tuple[int, int, int]:
        colors = {
            'ball': (0, 255, 255),  # Yellow
            'gkeep': (255, 0, 0),   # Blue
            'pl': (0, 255, 0),      # Green
            'ref': (255, 0, 255)    # Magenta
        }
        return colors.get(object_type, (255, 255, 255))  # White for unknown types
