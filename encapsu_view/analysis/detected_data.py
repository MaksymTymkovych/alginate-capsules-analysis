import math
from typing import Tuple, List, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt


from encapsu_view.entities.capsule import Capsule
from encapsu_view.entities.scale import Scale

from .processed_data import ProcessedData
from .data_processor import DataProcessor

class DetectedData:
    def __init__(self, capsules: List[Capsule], scale: Scale, image_shape: Tuple[int, int] = None):
        self.capsules = capsules
        self.scale = scale
        self.image_shape = image_shape

        if image_shape is not None:
            for capsule in capsules:
                capsule.is_complete = capsule.outer_contour.is_within_bounds(image_shape)
                if capsule.is_complete:
                    for inner in capsule.inner_objects:
                        if not inner.is_within_bounds(image_shape):
                            capsule.is_complete = False
                            break

        self.complete_capsules = [c for c in capsules if c.is_complete]
        self.cropped_capsules = [c for c in capsules if not c.is_complete]

    def __repr__(self):
        capsules_info = "\n  ".join([f"{capsule}" for capsule in self.capsules])
        complete_count = len(self.complete_capsules)
        cropped_count = len(self.cropped_capsules)
        
        return (f"DetectedData(\n"
                f"  scale={self.scale},\n"
                f"  image_shape={self.image_shape},\n"
                f"  capsules=[\n  {capsules_info}\n  ],\n"
                f"  summary: {complete_count} complete, {cropped_count} cropped\n)")

    def get_complete_capsules(self) -> List[Capsule]:
        return self.complete_capsules

    def get_cropped_capsules(self) -> List[Capsule]:
        return self.cropped_capsules

    def get_completeness_ratio(self) -> float:
        if not self.capsules:
            return 0.0
        return len(self.complete_capsules) / len(self.capsules)

    def visualize_detection(self, 
                          image: np.ndarray,
                          show_labels: bool = True,
                          show_bbox: bool = False,
                          capsule_color: Tuple[int, int, int] = (255, 0, 0),
                          inner_object_color: Tuple[int, int, int] = (0, 255, 0),
                          thickness: int = 2) -> np.ndarray:
        vis_image = image.copy()
        
        for capsule in self.capsules:
            if len(capsule.outer_contour.points) > 0:
                cv2.drawContours(vis_image, [capsule.outer_contour.points.astype(np.int32)], -1, capsule_color, thickness)
                
                if show_bbox:
                    x, y, w, h = capsule.outer_contour.bbox
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), capsule_color, 1)
                
                if show_labels:
                    x, y, w, h = capsule.outer_contour.bbox
                    status = "complete" if capsule.is_complete else "cropped"
                    label = f"Capsule {capsule.id} ({status})"
                    cv2.putText(vis_image, label, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, capsule_color, 1)
        
        for capsule in self.capsules:
            for inner_obj in capsule.inner_objects:
                if len(inner_obj.points) > 0:
                    cv2.drawContours(vis_image, [inner_obj.points.astype(np.int32)], -1, inner_object_color, thickness)
                    
                    if show_bbox:
                        x, y, w, h = inner_obj.bbox
                        cv2.rectangle(vis_image, (x, y), (x + w, y + h), inner_object_color, 1)
                    
                    if show_labels:
                        x, y, w, h = inner_obj.bbox
                        label = f"Inner {inner_obj.id}"
                        cv2.putText(vis_image, label, (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, inner_object_color, 1)
        
        return vis_image

    def plot_detection(self, 
                     image: np.ndarray,
                     figsize: Tuple[int, int] = (12, 8),
                     show_labels: bool = True,
                     show_bbox: bool = False,
                     capsule_color: Tuple[int,int,int] = (0,0,255),
                     inner_object_color: Tuple[int,int,int] = (0,255,0),
                     save_path: Optional[str] = None):
        vis_image = self.visualize_detection(image, show_labels, show_bbox)
        
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=figsize)
        plt.imshow(vis_image_rgb)
        plt.axis('off')
        
        complete_count = len(self.complete_capsules)
        cropped_count = len(self.cropped_capsules)
        total_inner = sum(len(capsule.inner_objects) for capsule in self.capsules)
        
        title = (f"Detected Capsules: {len(self.capsules)} "
                f"(Complete: {complete_count}, Cropped: {cropped_count})\n"
                f"Inner Objects: {total_inner}, Scale: {self.scale}")
        plt.title(title, fontsize=12)
        
        legend_elements = [
            plt.Line2D([0], [0], color=np.array(capsule_color)/255, lw=2, label='Capsules'),
            plt.Line2D([0], [0], color=np.array(inner_object_color)/255, lw=2, label='Inner Objects')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Image saved in: {save_path}")
        
        plt.tight_layout()
        plt.show()

    def visualize_detection_step_by_step(self, 
                                       image: np.ndarray,
                                       capsule_detector: callable,
                                       inner_detector: callable,
                                       figsize: Tuple[int, int] = (15, 5)):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax1.imshow(original_rgb)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        capsule_vis = image.copy()
        detected_capsules = capsule_detector(image)
        for contour in detected_capsules:
            if len(contour.points) > 0:
                cv2.drawContours(capsule_vis, [contour.points.astype(np.int32)], -1, (0, 0, 255), 2)
                if hasattr(contour, 'bbox'):
                    x, y, w, h = contour.bbox
                    cv2.putText(capsule_vis, contour.type, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        capsule_vis_rgb = cv2.cvtColor(capsule_vis, cv2.COLOR_BGR2RGB)
        ax2.imshow(capsule_vis_rgb)
        ax2.set_title(f'Capsule Detection: {len(detected_capsules)} found')
        ax2.axis('off')
        
        final_vis = self.visualize_detection(image, show_labels=True, show_bbox=False)
        final_vis_rgb = cv2.cvtColor(final_vis, cv2.COLOR_BGR2RGB)
        ax3.imshow(final_vis_rgb)
        ax3.set_title(f'Final Result: {len(self.capsules)} capsules')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.show()

    def create_detection_report(self, 
                              image: np.ndarray,
                              save_path: Optional[str] = None) -> np.ndarray:

        vis_image = self.visualize_detection(image, show_labels=True, show_bbox=True)
        
        height, width = vis_image.shape[:2]
        info_panel = np.ones((150, width, 3), dtype=np.uint8) * 255 
        
        y_offset = 30
        line_height = 25
        
        info_lines = [
            f"Detection Report",
            f"Total Capsules: {len(self.capsules)}",
            f"Complete: {len(self.complete_capsules)} | Cropped: {len(self.cropped_capsules)}",
            f"Completeness Ratio: {self.get_completeness_ratio():.2f}",
            f"Total Inner Objects: {sum(len(c.inner_objects) for c in self.capsules)}",
            f"Scale: {self.scale}",
            f"Image Shape: {self.image_shape}"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = y_offset + i * line_height
            cv2.putText(info_panel, line, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        report_image = np.vstack([vis_image, info_panel])
        
        if save_path:
            cv2.imwrite(save_path, report_image)
            print(f"Report saved in: {save_path}")
        
        return report_image

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
                'total_inner_objects': sum(len(capsule.inner_objects) for capsule in self.capsules)
            }
        }


    def convert_to_processed_data(self, ellipse_method: str = "direct", visual_debug: bool = False) -> ProcessedData:
        return DataProcessor.convert_to_processed_data(self, ellipse_method, visual_debug)

#DetectedData.convert_to_processed = convert_to_processed_data