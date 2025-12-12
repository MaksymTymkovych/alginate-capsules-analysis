from typing import List, Tuple, Callable

import numpy as np

from encapsu_view.entities.capsule import Capsule
from encapsu_view.entities.contour import Contour

class CapsuleAssembly:

    def __init__(self,
                 capsule_detector: Callable[[np.ndarray], List[Contour]],
                 inner_object_detector: Callable[[np.ndarray], List[Contour]],
                 matching_threshold: float = 0.8):
        self.capsule_detector = capsule_detector
        self.inner_object_detector = inner_object_detector
        self.matching_threshold = matching_threshold


    def process_image(self, image: np.ndarray) -> List[Capsule]:
        capsule_contours = self.capsule_detector(image)
        inner_contours = self.inner_object_detector(image)

        print(f"{len(capsule_contours)} capsules and {len(inner_contours)} inner objects found")

        for contour in capsule_contours:
            contour.type = "capsule"
        for contour in inner_contours:
            contour.type = "inner_object"

        capsules = self._assign_inner_objects_to_capsules(capsule_contours, inner_contours)
        
        return capsules

    def _assign_inner_objects_to_capsules(self, 
                                        capsule_contours: List[Contour],
                                        inner_contours: List[Contour]) -> List[Capsule]:
        capsules = []
        
        for i, capsule_contour in enumerate(capsule_contours):
            capsule_inner_objects = []
            
            for inner_contour in inner_contours:
                if self._is_object_inside_capsule(inner_contour, capsule_contour):
                    capsule_inner_objects.append(inner_contour)
            
            capsule = Capsule(
                outer_contour=capsule_contour,
                inner_objects=capsule_inner_objects,
                image_shape=None,  # Set it later
                capsule_id=f"capsule_{i}"
            )
            capsules.append(capsule)
            
            print(f"Capsule {capsule.id}: {len(capsule_inner_objects)} internal objects")
        
        return capsules

    def _is_object_inside_capsule(self, inner_contour: Contour, capsule_contour: Contour) -> bool:
        if inner_contour.is_inside(capsule_contour):
            return True

        if inner_contour.area < capsule_contour.area:
            inner_bbox = inner_contour.bbox
            capsule_bbox = capsule_contour.bbox

            iou = self._calculate_bbox_iou(inner_bbox, capsule_bbox)
            if iou > self.matching_threshold:
                return True
        
        return False

    def _calculate_bbox_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        union_area = w1 * h1 + w2 * h2 - inter_area

        if union_area == 0:
            return 0
        
        return inter_area / union_area
