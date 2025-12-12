import math
from typing import Tuple, List, Optional

import numpy as np
import cv2


class Contour:
    def __init__(self, points: np.ndarray, contour_id: Optional[str] = None, contour_type: str = "unknown"):
        self.points = points  # NumPy array with points [N, 2]
        self.id = contour_id or f"contour_{id(self)}"
        self.type = contour_type # "capsule", "inner_object"

        self.area = cv2.contourArea(points) if len(points) > 0 else 0
        self.bbox = cv2.boundingRect(points) if len(points) > 0 else (0, 0, 0, 0)
        self.centroid = self._calculate_centroid()


    def _calculate_centroid(self) -> Tuple[float, float]:
        if len(self.points) == 0:
            return (0, 0)
        M = cv2.moments(self.points)
        if M["m00"] == 0:
            return (self.points[0][0], self.points[0][1])
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        return (cx, cy)


    def is_inside(self, other: "Contour") -> bool:
        if len(self.points) == 0 or len(other.points) == 0:
            return False
        return cv2.pointPolygonTest(other.points, self.centroid, False) >= 0


    def is_within_bounds(self, image_shape: Tuple[int, int]) -> bool:
        if len(self.points) == 0:
            return True
            
        height, width = image_shape
        x_coords = self.points[:, 0]
        y_coords = self.points[:, 1]
        
        return (x_coords.min() >= 0 and x_coords.max() < width and 
                y_coords.min() >= 0 and y_coords.max() < height)


    def __repr__(self):
        if len(self.points) > 0:
            return f"Contour(id={self.id}, type={self.type}, points={self.points.shape}, area={self.area:.1f})"
        else:
            return f"Contour(id={self.id}, type={self.type}, empty)"

    def interpolate_to_max_distance(self, max_distance: float = 1.0) -> 'Contour':
        if len(self.points) < 2:
            return self
        
        new_points = []
        
        for i in range(len(self.points)):
            current_point = self.points[i]
            next_point = self.points[(i + 1) % len(self.points)]
            
            new_points.append(current_point)
            
            distance = np.linalg.norm(next_point - current_point)
            
            if distance > max_distance:
                num_segments = int(np.ceil(distance / max_distance))
                for t in np.linspace(0, 1, num_segments + 1)[1:-1]:
                    interpolated_point = current_point + t * (next_point - current_point)
                    new_points.append(interpolated_point)
        
        if len(new_points) > 0:
            new_points_array = np.array(new_points, dtype=np.float32).reshape(-1, 2)
        else:
            new_points_array = np.array([], dtype=np.float32).reshape(1, 2)
        
        return Contour(
            points=new_points_array,
            contour_id=f"{self.id}_interpolated",
            contour_type=self.type
        )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.type,
            'area': float(self.area),
            'centroid': [float(self.centroid[0]), float(self.centroid[1])],
            'bbox': {
                'x': float(self.bbox[0]),
                'y': float(self.bbox[1]),
                'width': float(self.bbox[2]),
                'height': float(self.bbox[3])
            },
            'num_points': len(self.points),
            'points': self.points.reshape(-1, 2).tolist() if len(self.points) > 0 else []
        }
