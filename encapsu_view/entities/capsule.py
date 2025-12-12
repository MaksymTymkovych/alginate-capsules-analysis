import math
from typing import Tuple, List, Optional

import numpy as np

from .contour import Contour


class Capsule:
    def __init__(self, outer_contour: Contour, inner_objects: List[Contour] = None, image_shape: Tuple[int, int] = None, capsule_id: Optional[str] = None):
        self.outer_contour = outer_contour
        self.inner_objects = inner_objects if inner_objects is not None else []
        self.id = capsule_id or f"capsule_{id(self)}"
        self.is_complete = True
        if image_shape is not None:
            if not outer_contour.is_within_bounds(image_shape):
                self.is_complete = False
            else:
                for inner_contour in self.inner_objects:
                    if not inner_contour.is_within_bounds(image_shape):
                        self.is_complete = False
                        break

    def add_inner_object(self, contour: Contour):
        self.inner_objects.append(contour)

    def __repr__(self):
        status = "complete" if self.is_complete else "cropped"
        return f"Capsule(id={self.id}, outer={self.outer_contour}, inners={len(self.inner_objects)}, status={status})"

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'is_complete': self.is_complete,
            'outer_contour': self.outer_contour.to_dict(),
            'inner_objects': [inner.to_dict() for inner in self.inner_objects],
            'num_inner_objects': len(self.inner_objects),
            'total_inner_area': sum(inner.area for inner in self.inner_objects)
        }
