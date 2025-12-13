from typing import Tuple, List, Dict, Optional

import cv2
import numpy as np

def create_color_range_from_bgr(bgr_color: Tuple[int, int, int], 
                              tolerance: int = 20) -> List[Dict[str, Tuple[int, int, int]]]:
    b, g, r = bgr_color
    
    bgr_array = np.uint8([[[b, g, r]]])
    hsv_color = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2])
    
    print(f"BGR: {bgr_color} -> HSV: {(h, s, v)}")
    
    if h < 10 or h > 170:
        lower1 = (0, max(0, s - tolerance), max(0, v - tolerance))
        upper1 = (min(180, 10 + tolerance), min(255, s + tolerance), min(255, v + tolerance))
        
        lower2 = (max(0, 170 - tolerance), max(0, s - tolerance), max(0, v - tolerance))
        upper2 = (180, min(255, s + tolerance), min(255, v + tolerance))
        
        print(f"Range 1: lower={lower1}, upper={upper1}")
        print(f"Range 2: lower={lower2}, upper={upper2}")
        
        return [
            {'lower': tuple(int(x) for x in lower1), 'upper': tuple(int(x) for x in upper1)},
            {'lower': tuple(int(x) for x in lower2), 'upper': tuple(int(x) for x in upper2)}
        ]
    else:
        lower = (max(0, h - tolerance), max(0, s - tolerance), max(0, v - tolerance))
        upper = (min(180, h + tolerance), min(255, s + tolerance), min(255, v + tolerance))
        
        print(f"Standard range: lower={lower}, upper={upper}")
        
        return [{'lower': tuple(int(x) for x in lower), 'upper': tuple(int(x) for x in upper)}]


def detect_lines_from_mask(mask: np.ndarray, min_line_length: int) -> List:
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=10
    )
    
    if lines is None:
        return []
    
    return lines.tolist()


def find_longest_horizontal_line(lines: List, min_line_length: int) -> Optional[Dict]:
    longest_line = None
    max_length = 0
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 10 or angle > 170:
            if length > max_length and length > min_line_length:
                max_length = length
                longest_line = {
                    'start': (x1, y1),
                    'end': (x2, y2),
                    'length': length,
                    'angle': angle
                }
    
    return longest_line