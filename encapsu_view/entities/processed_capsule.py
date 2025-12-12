import math
from typing import Tuple, List, Optional

import numpy as np

from .ellipse import Ellipse
from .contour import Contour



class ProcessedCapsule:
    def __init__(self, ellipse: Ellipse, inner_objects: List[Contour] = None, 
                 image_shape: Tuple[int, int] = None, capsule_id: Optional[str] = None):
        self.ellipse = ellipse
        self.inner_objects = inner_objects if inner_objects is not None else []
        self.id = capsule_id or f"processed_capsule_{id(self)}"

        self.is_complete = self._calculate_completeness(image_shape) if image_shape else False

    def _calculate_completeness(self, image_shape: Tuple[int, int], sample_points: int = 1000) -> bool:
        if not image_shape:
            return False
            
        height, width = image_shape
        points_inside = 0
        
        for _ in range(sample_points):
            while True:
                x = np.random.uniform(-1, 1)
                y = np.random.uniform(-1, 1)
                if x*x + y*y <= 1.0:
                    break
            
            angle_rad = math.radians(self.ellipse.angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            
            a, b = self.ellipse.axes[0] / 2, self.ellipse.axes[1] / 2
            x_ellipse = self.ellipse.center[0] + a * x * cos_a - b * y * sin_a
            y_ellipse = self.ellipse.center[1] + a * x * sin_a + b * y * cos_a
            
            if 0 <= x_ellipse < width and 0 <= y_ellipse < height:
                points_inside += 1
        
        ratio_inside = points_inside / sample_points
        return ratio_inside >= 0.75

    def __repr__(self):
        status = "complete" if self.is_complete else "cropped"
        return f"ProcessedCapsule(id={self.id}, ellipse={self.ellipse}, inners={len(self.inner_objects)}, status={status})"

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'is_complete': self.is_complete,
            'ellipse': self.ellipse.to_dict(),
            'inner_objects': [inner.to_dict() for inner in self.inner_objects],
            'num_inner_objects': len(self.inner_objects),
            'total_inner_area': sum(inner.area for inner in self.inner_objects),
            'filling_ratio': sum(inner.area for inner in self.inner_objects) / self.ellipse.area if self.ellipse.area > 0 else 0
        }
