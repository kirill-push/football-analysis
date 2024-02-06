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
