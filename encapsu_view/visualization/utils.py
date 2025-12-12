from typing import Tuple, List, Optional, Dict

import numpy as np
import cv2
import matplotlib.pyplot as plt


def visualize_raw_detections(image: np.ndarray,
                           capsule_detector: callable,
                           inner_detector: callable,
                           show_plot: bool = True) -> np.ndarray:

    vis_image = image.copy()
    

    detected_capsules = capsule_detector(image)
    for contour in detected_capsules:
        if len(contour.points) > 0:
            cv2.drawContours(vis_image, [contour.points.astype(np.int32)], -1, (0, 0, 255), 2)
            x, y, w, h = contour.bbox
            cv2.putText(vis_image, "capsule", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    detected_inner = inner_detector(image)
    for contour in detected_inner:
        if len(contour.points) > 0:
            cv2.drawContours(vis_image, [contour.points.astype(np.int32)], -1, (0, 255, 0), 2)
            x, y, w, h = contour.bbox
            cv2.putText(vis_image, "inner", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    if show_plot:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Raw Detections: {len(detected_capsules)} capsules, {len(detected_inner)} inner objects")
        plt.axis('off')
        plt.show()
    
    return vis_image


def visualize_processed_detection(self, 
                                image: np.ndarray,
                                show_labels: bool = True,
                                show_bbox: bool = False,
                                capsule_color: Tuple[int, int, int] = (255, 0, 0),
                                inner_object_color: Tuple[int, int, int] = (0, 255, 0),
                                thickness: int = 2) -> np.ndarray:

    vis_image = image.copy()
    
    for capsule in self.capsules:
        ellipse_points = capsule.ellipse.get_points().astype(np.int32)
        center = (int(capsule.ellipse.center[0]), int(capsule.ellipse.center[1]))
        axes = (int(capsule.ellipse.axes[0] / 2), int(capsule.ellipse.axes[1] / 2))
        angle = capsule.ellipse.angle
        
        cv2.ellipse(vis_image, center, axes, angle, 0, 360, capsule_color, thickness)
        
        if show_bbox:
            x, y, w, h = capsule.ellipse.get_bounding_rect()
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), capsule_color, 1)
        
        if show_labels:
            x, y, w, h = capsule.ellipse.get_bounding_rect()
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



def draw_detected_ruler_advanced(image: np.ndarray,
                                ruler_data: Dict,
                                line_color: Tuple[int, int, int] = (0, 255, 0),
                                text_color: Tuple[int, int, int] = (255, 0, 0),
                                show_scale_bar: bool = True) -> np.ndarray:
    result_image = image.copy()
    start = ruler_data['ruler_line']['start']
    end = ruler_data['ruler_line']['end']
    
    cv2.line(result_image, tuple(start), tuple(end), line_color, 3)
    
    cv2.drawMarker(result_image, tuple(start), line_color, 
                   cv2.MARKER_CROSS, 15, 2)
    cv2.drawMarker(result_image, tuple(end), line_color, 
                   cv2.MARKER_CROSS, 15, 2)
    
    if show_scale_bar and 'text' in ruler_data and ruler_data['text']:
        result_image = draw_scale_bar(result_image, ruler_data, line_color, text_color)
    
    result_image = add_ruler_info_text(result_image, ruler_data, text_color)
    
    return result_image

def draw_scale_bar(image: np.ndarray,
                  ruler_data: Dict,
                  line_color: Tuple[int, int, int],
                  text_color: Tuple[int, int, int]) -> np.ndarray:
    result_image = image.copy()
    start = ruler_data['ruler_line']['start']
    end = ruler_data['ruler_line']['end']
    
    is_horizontal = abs(start[1] - end[1]) < abs(start[0] - end[0])
    
    if is_horizontal:
        length = abs(end[0] - start[0])
        num_marks = 5
        for i in range(num_marks + 1):
            x = start[0] + (i * length) // num_marks
            y = start[1]
            cv2.line(result_image, (x, y - 10), (x, y + 10), line_color, 1)
    else:
        length = abs(end[1] - start[1])
        num_marks = 5
        for i in range(num_marks + 1):
            x = start[0]
            y = start[1] + (i * length) // num_marks
            cv2.line(result_image, (x - 10, y), (x + 10, y), line_color, 1)
    
    return result_image

def add_ruler_info_text(image: np.ndarray,
                       ruler_data: Dict,
                       text_color: Tuple[int, int, int]) -> np.ndarray:

    result_image = image.copy()
    start = ruler_data['ruler_line']['start']
    end = ruler_data['ruler_line']['end']
    
    mid_x = (start[0] + end[0]) // 2
    mid_y = (start[1] + end[1]) // 2
    
    lines = []
    
    lines.append(f"Pixels: {ruler_data['ruler_line']['length']:.1f}")
    
    if 'text' in ruler_data and ruler_data['text']:
        text_info = ruler_data['text']
        if text_info['value']:
            scale_um_per_pixel = text_info['value'] / ruler_data['ruler_line']['length']
            lines.append(f"Scale: {scale_um_per_pixel:.4f} {text_info['unit']}/px")
            lines.append(f"Real size: {text_info['value']} {text_info['unit']}")
    
    lines.append(f"Start: ({start[0]}, {start[1]})")
    lines.append(f"End: ({end[0]}, {end[1]})")
    
    text_x = min(start[0], end[0])
    text_y = min(start[1], end[1]) - 10
    
    if text_y < len(lines) * 25:
        text_y = max(start[1], end[1]) + 20
    
    for i, line in enumerate(lines):
        y_pos = text_y + i * 25
        cv2.putText(result_image, line, (text_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    return result_image



def visualize_ruler_detection(image: np.ndarray, ruler_data: Dict, save_path: Optional[str] = None):
    visualized_image = draw_detected_ruler_advanced(image, ruler_data)
    
    visualized_image_rgb = cv2.cvtColor(visualized_image, cv2.COLOR_BGR2RGB)
    plt.imshow(visualized_image_rgb)
    plt.show()
    #cv2.imshow('Detected Ruler', visualized_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    if save_path:
        cv2.imwrite(save_path, visualized_image)
        print(f"Image saved in: {save_path}")
    
    return visualized_image
