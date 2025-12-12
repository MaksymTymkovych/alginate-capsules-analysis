import math
from typing import Tuple

import numpy as np
import cv2

class Ellipse:
    def __init__(self, center: Tuple[float, float], axes: Tuple[float, float], angle: float):
        self.center = center
        self.axes = axes
        self.angle = angle

        self.area = math.pi * self.axes[0] * self.axes[1]  / 4

    def is_point_inside(self, point: Tuple[float, float]) -> bool:
        x, y = point
        cx, cy = self.center

        angle_rad = math.radians(self.angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        dx = x - cx
        dy = y - cy

        x_rot = dx * cos_a + dy * sin_a
        y_rot = -dx * sin_a + dy * cos_a

        a, b = self.axes[0] / 2, self.axes[1] / 2
        return (x_rot * x_rot) / (a * a) + (y_rot * y_rot) / (b * b) <= 1.0

    def get_bounding_rect(self) -> Tuple[float, float, float, float]:
        return cv2.boundingRect(self.get_points())

    def get_points(self, num_points: int = 100) -> np.ndarray:
        points = []
        for i in range(num_points):
            theta = 2 * math.pi * i / num_points
            angle_rad = math.radians(self.angle)
            
            x = self.center[0] + (self.axes[0] / 2) * math.cos(theta) * math.cos(angle_rad) - \
                (self.axes[1] / 2) * math.sin(theta) * math.sin(angle_rad)
            y = self.center[1] + (self.axes[0] / 2) * math.cos(theta) * math.sin(angle_rad) + \
                (self.axes[1] / 2) * math.sin(theta) * math.cos(angle_rad)
            points.append([x, y])
        
        return np.array(points, dtype=np.float32)

    def __repr__(self):
        return f"Ellipse(center={self.center}, axes={self.axes}, angle={self.angle:.1f}°, area={self.area:.1f})"

    def to_dict(self) -> dict:
        return {
            'center': [float(self.center[0]), float(self.center[1])],
            'axes': [float(self.axes[0]), float(self.axes[1])],
            'angle': float(self.angle),
            'area': float(self.area),
            'major_axis': float(max(self.axes[0], self.axes[1])),
            'minor_axis': float(min(self.axes[0], self.axes[1])),
            'aspect_ratio': float(max(self.axes[0], self.axes[1]) / min(self.axes[0], self.axes[1])) if min(self.axes[0], self.axes[1]) > 0 else 0
        }
