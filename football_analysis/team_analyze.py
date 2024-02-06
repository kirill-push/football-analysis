from typing import List, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

def find_teams(
    image: np.ndarray, boxes: List, del_bg: bool = True
) -> Tuple[List, List, np.ndarray]:
    """Finds and clusters football players into two teams based on their average colors.

    Args:
        image (np.ndarray): The input image in BGR.
        boxes (List): A list of bounding boxes for each football player.

    Returns:
        Tuple[List, List, np.ndarray]: A tuple containing two lists of player boxes for
            each team and the team colors.
    """

    # Get the average color for each box
    if del_bg:
        average_colors = [
            remove_background_and_get_avg_color(image, box)[1] for box in boxes
        ]
    else:
        average_colors = [get_average_color(image, box) for box in boxes]
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=0).fit(average_colors)
    labels = kmeans.labels_

    # Determine the average color for each team
    team_colors = kmeans.cluster_centers_

    # Assign players to teams based on labels
    team_1 = [boxes[i] for i in range(len(boxes)) if labels[i] == 0]
    team_2 = [boxes[i] for i in range(len(boxes)) if labels[i] == 1]

    return team_1, team_2, team_colors.astype(int)
