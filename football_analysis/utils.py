from typing import List

import numpy as np


def get_average_color(image: np.ndarray, box: List) -> np.ndarray:
    """Calculates average color of a region within an image specified by bounding box.

    Args:
        image (np.ndarray): The input image in BGR.
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
    hands: bool = True,
) -> np.ndarray:
    """Crops an image based on a bounding box and optionally retains specified
    percentages of the head, legs, and hands.

    Args:
        image (np.ndarray): The input image in BGR.
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
            int(width * 0.25) : int(width * (1 - 0.25)),
        ]
    else:
        return crop_img[head_pixels : height - legs_pixels, :]
