from typing import List, Tuple, Callable, Dict

import numpy as np

from .entities import Capsule, Contour
from detected_data import DetectedData


def process_image_with_capsules(image: np.ndarray,
                               capsule_detector: Callable[[np.ndarray], List[Contour]],
                               inner_object_detector: Callable[[np.ndarray], List[Contour]],
                               scale_detector: Callable[[np.ndarray], Dict]) -> DetectedData:

    """
    ruler_color = (0, 0, 255) # BGR
    ruler_color_range = create_color_range_from_bgr(ruler_color, tolerance=10)
    ruler = detect_scale_ruller(image, ruler_color_range)

    scale = None
    if ruler:
        scale = ruler_data_to_scale(ruler)    
    """
    scale_data = scale_detector(image)
    #scale = ruler_data_to_scale(ruler)

    assembler = CapsuleAssembly(capsule_detector, inner_object_detector)
    capsules = assembler.process_image(image)

    return DetectedData(capsules, scale_data, image.shape[:2])