import json
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

from football_analysis.utils import get_average_color


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
        self.n_frames = len(self.frames)

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

    def preprocess_bboxes(
        self,
        min_width: float,
        max_proportion: float,
        min_proportion: float,
        min_area: float,
    ) -> None:
        self._calculate_bboxes_stats("pl")
        for frame_idx, frame_bboxes in self.bboxes_stats.items():
            for bbox_idx, bbox_stats in frame_bboxes.items():
                width = bbox_stats["width"]
                proportion = bbox_stats["proportion"]
                area = bbox_stats["area"]

                # Check if bbox has bad params
                if (
                    width < min_width
                    or proportion > max_proportion
                    or proportion < min_proportion
                    or area < min_area
                ):
                    self.bboxes_stats[frame_idx][bbox_idx]["class"] = None
                else:
                    self.bboxes_stats[frame_idx][bbox_idx]["class"] = -1

    def calculate_bboxes_stats(self, bbox_type: str = "pl") -> None:
        bboxes = self.bboxes[bbox_type]
        self.bboxes_stats: Dict[int, Dict[int, Dict[str, Any]]] = {}
        self.bboxes_areas: List[float] = []
        self.bboxes_proportions: List[float] = []
        self.bboxes_ids = []
        self.bboxes_conf = []

        for i, frame_boxes in enumerate(bboxes):
            self.bboxes_stats[i] = dict()
            for j, bbox in enumerate(frame_boxes):
                self.bboxes_ids.append((i, j))
                self.bboxes_conf.append(bbox[-1])
                # Calculate area: (x2 - x1) * (y2 - y1)
                height = bbox[3] - bbox[1]
                width = bbox[2] - bbox[0]
                area = width * height
                self.bboxes_areas.append(area)

                proportion = height / width
                self.bboxes_proportions.append(proportion)

                self.bboxes_stats[i][j] = {
                    "height": height,
                    "width": width,
                    "area": area,
                    "proportion": proportion,
                    "bbox": bbox,
                    "confidence": bbox[-1],
                }

    def assign_teams_to_bboxes(self, del_bg: bool = True) -> None:
        """
        Assigns bboxes to two teams across all frames, ensuring consistency in team colors.
        """
        self.color_history = []
        for frame_idx, frame in enumerate(self.frames):
            # Filter bboxes for this frame with class -1 (valid bboxes)
            valid_bboxes = [
                self.bboxes_stats[frame_idx][bbox_idx]["bbox"]
                for bbox_idx in self.bboxes_stats[frame_idx]
                if self.bboxes_stats[frame_idx][bbox_idx]["class"] is not None
            ]

            # If there are valid bboxes to process
            if valid_bboxes:
                team_1, team_2, team_colors = find_teams(
                    frame, valid_bboxes, del_bg=del_bg
                )
                if frame_idx > 0:
                    team_1, team_2, team_colors, last_colors = (
                        self.compare_color_with_previous(
                            team_1, team_2, team_colors, last_colors
                        )
                    )
                    self.team_colors = (self.team_colors * frame_idx + team_colors) / (
                        frame_idx + 1
                    )
                else:
                    last_colors = team_colors.copy()
                    self.team_colors = team_colors.copy()
                self.color_history.append(team_colors)
                # Update bbox classes based on team assignment
                for bbox_idx, bbox_stats in self.bboxes_stats[frame_idx].items():
                    if (
                        bbox_stats["class"] == -1
                    ):  # Process only previously valid bboxes
                        bbox = bbox_stats["bbox"]
                        if bbox in team_1:
                            self.bboxes_stats[frame_idx][bbox_idx]["class"] = 0
                        elif bbox in team_2:
                            self.bboxes_stats[frame_idx][bbox_idx]["class"] = 1

    def compare_color_with_previous(self, team_1, team_2, team_colors, last_colors):
        if np.linalg.norm(team_colors[0] - last_colors[1]) + np.linalg.norm(
            team_colors[1] - last_colors[0]
        ) < np.linalg.norm(team_colors[0] - last_colors[0]) + np.linalg.norm(
            team_colors[1] - last_colors[1]
        ):
            # Если цвета команд лучше соответствуют в инверсии, меняем местами
            return team_2, team_1, team_colors[::-1], team_colors[::-1]
        return team_1, team_2, team_colors, team_colors

    def get_item(self, frame_idx: int) -> Dict[str, Any] | None:
        if frame_idx < 0 or frame_idx >= len(self.frames):
            return None
        frame_data: Dict[str, np.ndarray | Dict] = {"frame": self.frames[frame_idx]}
        frame_data["bboxes"] = {}
        for key, bboxes_list in self.bboxes.items():
            if frame_idx < len(bboxes_list):
                frame_data["bboxes"][key] = bboxes_list[frame_idx]
        return frame_data

    def find_colors(self, eps: float = 10, min_samples: int | None = None) -> None:
        average_colors = []
        for i, frame_stat in self.bboxes_stats.items():
            average_colors.extend(
                [
                    get_average_color(self.frames[i], bbox_stat["bbox"])
                    for bbox_stat in frame_stat.values()
                    if bbox_stat["class"] is not None
                ]
            )
        # Perform DBSCAN clustering
        if min_samples is None:
            min_samples = len(self.frames)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(average_colors)

        labels_dbscan = dbscan.labels_
        if len(labels_dbscan) != 3:
            print(
                f"Try to use other params for DBSCAN, it was found {len(labels_dbscan) - 1} clusters"
            )
        team_0_labels = np.where(labels_dbscan == 0)[0]
        team_1_labels = np.where(labels_dbscan == 1)[0]

        team_0_colors = np.array(average_colors)[team_0_labels]
        team_1_colors = np.array(average_colors)[team_1_labels]

        team_0_color = np.mean(team_0_colors, 0)
        team_1_color = np.mean(team_1_colors, 0)

        self.team_colors = np.vstack((team_0_color, team_1_color))

    def match_bbox_to_color(self) -> None:
        for i, frame_stat in self.bboxes_stats.items():
            for j, bbox_stat in frame_stat.items():
                if bbox_stat["class"] is not None:
                    average_color = get_average_color(self.frames[i], bbox_stat["bbox"])
                    if self.color_distance(
                        average_color, self.team_colors[0]
                    ) < self.color_distance(average_color, self.team_colors[1]):
                        self.bboxes_stats[i][j]["class"] = 0
                    else:
                        self.bboxes_stats[i][j]["class"] = 1

    def color_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        weight = np.array([0.11, 0.59, 0.3])
        return np.sqrt(np.sum(weight * (x - y) * (x - y)))

    def visualize_frame(
        self,
        frame_idx: int,
        box_type: str | None = None,
        bboxes_list: List | None = None,
        draw_team: bool = False,
    ) -> None:
        if not draw_team:
            frame_data = self.get_item(frame_idx)
            if frame_data is None:
                print("Frame index out of bounds.")
                return

            frame = frame_data["frame"].copy()
            plt.figure(figsize=(30, 3))
            if bboxes_list is None:
                for key, bboxes in frame_data["bboxes"].items():
                    for bbox in bboxes:
                        if box_type is None or key == box_type:
                            x1, y1, x2, y2, _ = bbox
                            cv2.rectangle(
                                frame,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                self.get_color_by_type(key),
                                2,
                            )
            else:
                if box_type is None:
                    box_type = "pl"
                for bbox in bboxes_list:
                    x1, y1, x2, y2, _ = bbox
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        self.get_color_by_type(box_type),
                        2,
                    )

            # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # plt.axis("off")
            # plt.show()
            return frame

        else:
            frame = self.frames[frame_idx].copy()
            for bbox_stats in self.bboxes_stats[frame_idx].values():
                if bbox_stats["class"] in [0, 1]:  # Draw only valid team members
                    bbox = bbox_stats["bbox"]
                    color = self.team_colors[bbox_stats["class"]]
                    x1, y1, x2, y2 = [int(coord) for coord in bbox[:4]]
                    team_label = f"Team {bbox_stats['class']}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        team_label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

            return frame

    @staticmethod
    def get_color_by_type(object_type: str) -> Tuple[int, int, int]:
        colors = {
            "ball": (0, 255, 255),  # Yellow
            "gkeep": (255, 0, 0),  # Blue
            "pl": (0, 255, 0),  # Green
            "ref": (255, 0, 255),  # Magenta
        }
        return colors.get(object_type, (255, 255, 255))  # White for unknown types
