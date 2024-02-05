from typing import List, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans


def remove_background_and_get_avg_color(
    image: np.ndarray, box: List, head: float = 0.2, legs: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Removes the background from an image of a football player and calculates average
        color of the player.

    Args:
        image (np.ndarray): The input image.
        box (List): A list containing the coordinates of the bounding box in format:
            (x1, y1, x2, y2, _).
        head (float): The percentage of the player's head to retain.
            Defaults to 0.2
        legs (float): The percentage of the player's legs to retain.
            Defaults to 0.5

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the masked image and average
            color of the player.
    """

    # Crop the football player
    crop_img = crop_image(image, box, head, legs)

    # Convert the image to HSV
    hsv_crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2HSV)

    # Define the range for green color
    # TODO: These values may need adjustment
    lower_green = np.array([35, 25, 25])
    upper_green = np.array([85, 255, 255])

    # Create a mask for green color
    mask = cv2.inRange(hsv_crop_img, lower_green, upper_green)

    # Invert the mask to make green color the background
    mask_inv = cv2.bitwise_not(mask)

    # Apply the mask to the image
    masked_img = cv2.bitwise_and(crop_img, crop_img, mask=mask_inv)

    non_zero_pixels = np.any(masked_img != [0, 0, 0], axis=-1)

    # Filter the image array, keeping only non-zero pixels
    filtered_img = masked_img[non_zero_pixels]

    # Calculate the average color of the image without (0, 0, 0)
    avg_color = np.mean(filtered_img, axis=(0))

    return masked_img, avg_color


def get_average_color(image: np.ndarray, box: List) -> np.ndarray:
    """Calculates average color of a region within an image specified by bounding box.

    Args:
        image (np.ndarray): The input image.
        box (List): A list containing the coordinates of the bounding box in format
            (x1, y1, x2, y2, _).

    Returns:
        np.ndarray: The average color of the specified region.
    """
    crop_img = crop_image(image, box, head=0.2, legs=0.5)
    avg_color_per_row = np.average(crop_img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    return avg_color


def crop_image(
    image: np.ndarray,
    box: List,
    head: float = 0.2,
    legs: float = 0.5,
    hands: bool = False,
) -> np.ndarray:
    """Crops an image based on a bounding box and optionally retains specified
    percentages of the head, legs, and hands.

    Args:
        image (np.ndarray): The input image.
        box (List): A list containing the coordinates of the bounding box in format
            (x1, y1, x2, y2, _).
        head (float): The percentage of the player's head to retain.
        legs (float): The percentage of the player's legs to retain.
        hands (bool): Whether to retain the player's hands.

    Returns:
        np.ndarray: The cropped image.
    """

    (x1, y1, x2, y2, _) = box

    # Crop the image
    crop_img = image[int(y1) : int(y2), int(x1) : int(x2)]

    # Get the height and width of the image
    height = crop_img.shape[0]
    width = crop_img.shape[1]

    # Calculate the number of pixels to crop from the top and bottom
    head_pixels = int(height * head)
    legs_pixels = int(height * legs)
    if hands:
        return crop_img[
            head_pixels : height - legs_pixels,
            int(width * 0.2) : int(width * (1 - 0.2)),
        ]
    else:
        return crop_img[head_pixels : height - legs_pixels, :]


def find_teams(image: np.ndarray, boxes: List) -> Tuple[List, List, np.ndarray]:
    """Finds and clusters football players into two teams based on their average colors.

    Args:
        image (np.ndarray): The input image.
        boxes (List): A list of bounding boxes for each football player.

    Returns:
        Tuple[List, List, np.ndarray]: A tuple containing two lists of player boxes for
            each team and the team colors.
    """

    # Get the average color for each box
    average_colors = [
        remove_background_and_get_avg_color(image, box)[1] for box in boxes
    ]

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=0).fit(average_colors)
    labels = kmeans.labels_

    # Determine the average color for each team
    team_colors = kmeans.cluster_centers_

    # Assign players to teams based on labels
    team_1 = [boxes[i] for i in range(len(boxes)) if labels[i] == 0]
    team_2 = [boxes[i] for i in range(len(boxes)) if labels[i] == 1]

    return team_1, team_2, team_colors.astype(int)
