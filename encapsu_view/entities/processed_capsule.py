import math
import os
from typing import Tuple, List, Optional
import csv
import numpy as np
import cv2

from .ellipse import Ellipse
from .contour import Contour
from .scale import Scale


# Number of points sampled on the capsule ellipse to measure the distance to the core
SHELL_SAMPLE_POINTS = 360


class ProcessedCapsule:
    def __init__(self, ellipse: Ellipse, inner_objects: List[Contour] = None,
                 image_shape: Tuple[int, int] = None, capsule_id: Optional[str] = None):
        self.ellipse = ellipse
        self.inner_objects = inner_objects if inner_objects is not None else []
        self.id = capsule_id or f"processed_capsule_{id(self)}"

        self.is_complete = self._calculate_completeness(image_shape) if image_shape else False
        # All values are in pixels; conversion to microns happens in to_row()
        self.capsule_area=0
        self.core_area=0
        self.ratio=0
        self.form_factor=0
        self.capsule_radius_min=0
        self.capsule_radius_max=0
        self.core_radius_min=0
        self.core_radius_max=0
        self.core_circularity=0
        self.distance_to_core_min=0
        self.distance_to_core_max=0
        self.distance_to_core_median=0
        self.distance_to_core_mean=0
        self._calculate_parameters()

    def _calculate_parameters(self):
        self.capsule_area=self.ellipse.area
        # Ellipse axes are full lengths (diameters), radius is a half of it
        self.capsule_radius_min=min(self.ellipse.axes[0], self.ellipse.axes[1]) / 2
        self.capsule_radius_max=max(self.ellipse.axes[0], self.ellipse.axes[1]) / 2
        self.form_factor=self.capsule_radius_min/self.capsule_radius_max if self.capsule_radius_max > 0 else 0

        cores = self._valid_cores()

        self.core_area=0
        for contour in cores:
            self.core_area += contour.area

        self.ratio=self.core_area/self.capsule_area if self.capsule_area > 0 else 0

        if not cores:
            # Nothing to measure: leave core-related values empty instead of a misleading zero
            for name in ['core_radius_min', 'core_radius_max', 'core_circularity',
                         'distance_to_core_min', 'distance_to_core_max',
                         'distance_to_core_median', 'distance_to_core_mean']:
                setattr(self, name, None)
            return

        # Radii and circularity describe the main (largest) core
        main_core = max(cores, key=lambda c: c.area)
        points = main_core.points.reshape(-1, 2).astype(np.float64)
        radii = np.linalg.norm(points - np.array(main_core.centroid, dtype=np.float64), axis=1)
        self.core_radius_min=float(radii.min())
        self.core_radius_max=float(radii.max())

        perimeter = cv2.arcLength(main_core.points.reshape(-1, 1, 2).astype(np.float32), True)
        self.core_circularity=4 * math.pi * main_core.area / perimeter**2 if perimeter > 0 else 0

        distances = self._distances_to_core(cores)
        self.distance_to_core_min=float(distances.min())
        self.distance_to_core_max=float(distances.max())
        self.distance_to_core_median=float(np.median(distances))
        self.distance_to_core_mean=float(distances.mean())

    def _valid_cores(self) -> List[Contour]:
        return [c for c in self.inner_objects if len(c.points) >= 3 and c.area > 0]

    def _distances_to_core(self, cores: List[Contour]) -> np.ndarray:
        """Shell thickness: distance from each point of the capsule ellipse to the nearest core boundary."""
        core_polygons = [c.points.reshape(-1, 1, 2).astype(np.float32) for c in cores]
        distances = []
        for x, y in self.ellipse.get_points(SHELL_SAMPLE_POINTS):
            # pointPolygonTest is negative outside the polygon; a point inside a core means zero thickness
            nearest = min(-cv2.pointPolygonTest(polygon, (float(x), float(y)), True) for polygon in core_polygons)
            distances.append(max(0.0, nearest))
        return np.array(distances)

    def _calculate_completeness(self, image_shape: Tuple[int, int], sample_points: int = 1000) -> bool:
        if not image_shape:
            return False

        height, width = image_shape
        points_inside = 0
        # Fixed seed: the same capsule must always get the same status
        rng = np.random.default_rng(0)

        for _ in range(sample_points):
            while True:
                x = rng.uniform(-1, 1)
                y = rng.uniform(-1, 1)
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
            'capsule_area': self.capsule_area,
            'core_area': self.core_area,
            'ratio': self.ratio,
            'form_factor': self.form_factor,
            'capsule_radius_min': self.capsule_radius_min,
            'capsule_radius_max': self.capsule_radius_max,
            'core_radius_min': self.core_radius_min,
            'core_radius_max': self.core_radius_max,
            'core_circularity': self.core_circularity,
            'distance_to_core_min': self.distance_to_core_min,
            'distance_to_core_max': self.distance_to_core_max,
            'distance_to_core_median': self.distance_to_core_median,
            'distance_to_core_mean': self.distance_to_core_mean,
            'is_complete': self.is_complete,
            'ellipse': self.ellipse.to_dict(),
            'inner_objects': [inner.to_dict() for inner in self.inner_objects],
            'num_inner_objects': len(self.inner_objects),
            'total_inner_area': sum(inner.area for inner in self.inner_objects),
            'filling_ratio': sum(inner.area for inner in self.inner_objects) / self.ellipse.area if self.ellipse.area > 0 else 0
        }

    def to_row(self, scale: Optional[Scale] = None) -> dict:
        """Flat measurements for a table: lengths and areas in px and, if the scale is known, in µm."""
        ppm = scale.pixels_per_micron if scale is not None and scale.pixels_per_micron else None

        def length(value):
            return value / ppm if ppm and value is not None else None

        def area(value):
            return value / ppm**2 if ppm and value is not None else None

        row = {
            'capsule_id': self.id,
            'is_complete': self.is_complete,
            'num_cores': len(self._valid_cores()),
            'center_x_px': float(self.ellipse.center[0]),
            'center_y_px': float(self.ellipse.center[1]),
            'form_factor': self.form_factor,
            'core_circularity': self.core_circularity,
            'ratio': self.ratio,
        }

        for name in ['capsule_area', 'core_area']:
            value = getattr(self, name)
            row[f'{name}_px2'] = value
            row[f'{name}_um2'] = area(value)

        for name in ['capsule_radius_min', 'capsule_radius_max',
                     'core_radius_min', 'core_radius_max',
                     'distance_to_core_min', 'distance_to_core_max',
                     'distance_to_core_median', 'distance_to_core_mean']:
            value = getattr(self, name)
            row[f'{name}_px'] = value
            row[f'{name}_um'] = length(value)

        row['pixels_per_micron'] = ppm
        return row

    def save_to_csv(self, filename: str, overwrite: bool = True,
                    field_date: str = "", field_conditions: str = "",
                    field_image_name: str = "", scale: Optional[Scale] = None):
        row = {
            'date': field_date,
            'conditions': field_conditions,
            'image_name': field_image_name,
            **self.to_row(scale)
        }
        write_csv_rows(filename, [row], overwrite=overwrite)


def write_csv_rows(filename: str, rows: List[dict], overwrite: bool = True):
    """Write rows to CSV; when appending, the header is written only if the file is new or empty."""
    if not rows:
        return

    mode = 'w' if overwrite else 'a'
    write_header = overwrite or not os.path.exists(filename) or os.path.getsize(filename) == 0

    # utf-8-sig so that Excel shows non-ASCII values (e.g. conditions in Ukrainian) correctly
    encoding = 'utf-8-sig' if write_header else 'utf-8'
    with open(filename, mode=mode, newline='', encoding=encoding) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
