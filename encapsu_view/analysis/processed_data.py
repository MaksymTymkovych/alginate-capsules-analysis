import math
from typing import Tuple, List, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt


from encapsu_view.entities.capsule import Capsule
from encapsu_view.entities.scale import Scale
from encapsu_view.entities.processed_capsule import ProcessedCapsule

from encapsu_view.visualization.utils import visualize_processed_detection


class ProcessedData:  
    def __init__(self, capsules: List[ProcessedCapsule], scale: Scale, image_shape: Tuple[int, int] = None):
        self.capsules = capsules
        self.scale = scale
        self.image_shape = image_shape
        
        self.complete_capsules = [c for c in capsules if c.is_complete]
        self.cropped_capsules = [c for c in capsules if not c.is_complete]
    
    def __repr__(self):
        capsules_info = "\n  ".join([f"{capsule}" for capsule in self.capsules])
        complete_count = len(self.complete_capsules)
        cropped_count = len(self.cropped_capsules)
        
        return (f"ProcessedData(\n"
                f"  scale={self.scale},\n"
                f"  image_shape={self.image_shape},\n"
                f"  capsules=[\n  {capsules_info}\n  ],\n"
                f"  summary: {complete_count} complete, {cropped_count} cropped\n)")
    
    def get_complete_capsules(self) -> List[ProcessedCapsule]:
        return self.complete_capsules
    
    def get_cropped_capsules(self) -> List[ProcessedCapsule]:
        return self.cropped_capsules
    
    def get_completeness_ratio(self) -> float:
        if not self.capsules:
            return 0.0
        return len(self.complete_capsules) / len(self.capsules)

    def to_dict(self) -> dict:
        return {
            'image_shape': {
                'height': self.image_shape[0] if self.image_shape else 0,
                'width': self.image_shape[1] if self.image_shape else 0
            },
            'scale': {
                'pixels_per_micron': float(self.scale.pixels_per_micron),
                'unit': self.scale.unit
            },
            'completeness_ratio': float(self.get_completeness_ratio()),
            'capsules': [capsule.to_dict() for capsule in self.capsules],
            'statistics': {
                'total_capsules': len(self.capsules),
                'complete_capsules': len(self.complete_capsules),
                'cropped_capsules': len(self.cropped_capsules),
                'total_inner_objects': sum(len(capsule.inner_objects) for capsule in self.capsules),
                'average_ellipse_area': np.mean([capsule.ellipse.area for capsule in self.capsules]) if self.capsules else 0,
                'average_aspect_ratio': np.mean([max(capsule.ellipse.axes) / min(capsule.ellipse.axes) for capsule in self.capsules if min(capsule.ellipse.axes) > 0]) if self.capsules else 0
            }
        }

ProcessedData.visualize_detection = visualize_processed_detection
