from typing import List, Tuple

import cv2
import numpy as np


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
