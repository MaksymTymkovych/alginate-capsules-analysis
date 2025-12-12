from typing import Tuple, List, Dict, Optional

import cv2
import numpy as np

import pytesseract
import re

import matplotlib.pyplot as plt

from encapsu_view.entities.scale import Scale
from encapsu_view.analysis.utils import create_color_range_from_bgr, detect_lines_from_mask, find_longest_horizontal_line


def detect_scale_ruller(image: np.ndarray, 
                      color_ranges: List[Dict[str, Tuple[int, int, int]]],
                      min_line_length: int = 100,
                      text_region_height: int = 150) -> Optional[Dict]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for color_range in color_ranges:
        lower = np.array(color_range['lower'])
        upper = np.array(color_range['upper'])
        range_mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_or(mask, range_mask)

    plt.imshow(mask)
    plt.show()

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    lines = detect_lines_from_mask(mask, min_line_length)
    
    if not lines:
        return None

    ruler_line = find_longest_horizontal_line(lines, min_line_length)
    
    if ruler_line is None:
        return None

    text_info = extract_text_from_ruler_region(image, mask, ruler_line, text_region_height)
    
    
    return {
        'ruler_line': ruler_line,
        'text': text_info,
        'mask': mask
    }


def parse_scale_text(text: str) -> Tuple[Optional[float], Optional[str]]:
    clean_text = re.sub(r'[^\dumµ.]', '', text)
    
    match = re.search(r'(\d*\.?\d+)\s*([umµ]+)?', clean_text)
    
    if match:
        value = float(match.group(1))
        unit = match.group(2) if match.group(2) else 'um'
        return value, unit
    
    return None, None


def contains_numbers_and_units(text: str) -> bool:
    has_numbers = bool(re.search(r'\d', text))
    
    units = ['um', 'µm', 'mm', 'cm', 'm', 'µ']
    has_units = any(unit in text.lower() for unit in units)
    
    return has_numbers and has_units


def select_best_text_candidate(candidates: list) -> str:
    if not candidates:
        return ""
    
    for candidate in candidates:
        if contains_numbers_and_units(candidate):
            return candidate
    
    for candidate in candidates:
        if candidate.strip():
            return candidate
    
    return ""


def extract_text_from_ruler_region(image: np.ndarray, 
                                 mask: np.ndarray,
                                 ruler_line: Dict,
                                 region_height: int) -> Optional[Dict]:
    start = ruler_line['start']
    end = ruler_line['end']
    
    y_min = min(start[1], end[1]) - region_height
    y_max = min(start[1], end[1])
    x_min = min(start[0], end[0])
    x_max = max(start[0], end[0])
    
    y_min = max(0, y_min)
    y_max = max(0, y_max)
    x_min = max(0, x_min)
    x_max = min(image.shape[1], x_max)
    
    if y_min >= y_max or x_min >= x_max:
        return None
    
    text_region = image[y_min:y_max, x_min:x_max]
    text_mask_region = mask[y_min:y_max, x_min:x_max]
    
    if text_region.size == 0:
        return None

    text_processed = cv2.bitwise_and(text_region, text_region, mask=text_mask_region)
    text_gray = cv2.cvtColor(text_processed, cv2.COLOR_BGR2GRAY)
    _, text_binary = cv2.threshold(text_gray, 1, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2, 2), np.uint8)
    text_binary = cv2.morphologyEx(text_binary, cv2.MORPH_CLOSE, kernel)


    try:
        text_candidates = []
        custom_config1 = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789umµ '
        text1 = pytesseract.image_to_string(text_binary, config=custom_config1).strip()
        if text1:
            text_candidates.append(text1)

        custom_config2 = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789umµ '
        text2 = pytesseract.image_to_string(text_binary, config=custom_config2).strip()
        if text2 and text2 != text1:
            text_candidates.append(text2)

        custom_config3 = r'--oem 3 --psm 7'
        text3 = pytesseract.image_to_string(text_binary, config=custom_config3).strip()
        if text3 and text3 not in [text1, text2]:
            text_candidates.append(text3)


        best_text = select_best_text_candidate(text_candidates)

        
        #plt.imshow(text_binary)
        #plt.show()
        value, unit = parse_scale_text(best_text)
        #print(best_text)
        #print(value)
        #print(unit)
        return {
            'raw_text': best_text,
            'value': value,
            'unit': unit,
            'region': (x_min, y_min, x_max - x_min, y_max - y_min),
            'processed_image': text_binary,
            'start': start,
            'end': end
        }
    except Exception as e:
        print(f"Text recognition error: {e}")

    return None

def ruler_data_to_scale(ruler_data: dict) -> Scale:
    pixel_length = ruler_data['ruler_line']['length']
    
    text_info = ruler_data.get('text', {})
    real_length = text_info.get('value')
    unit = text_info.get('unit', 'um')
    
    if real_length is None or real_length <= 0:
        raise ValueError("Could not determine the actual length of the line from the text")
    
    if pixel_length <= 0:
        raise ValueError("The length of the line in pixels must be greater than 0")
    
    unit = unit.lower()
    if unit in ['um', 'µm', 'micrometer', 'micrometre']:
        real_length_microns = real_length
    elif unit in ['mm', 'millimeter', 'millimetre']:
        real_length_microns = real_length * 1000
    elif unit in ['cm', 'centimeter', 'centimetre']:
        real_length_microns = real_length * 10000
    elif unit in ['m', 'meter', 'metre']:
        real_length_microns = real_length * 1000000
    else:
        print(f"Warning: unknown unit of measurement ‘{unit}’. We assume it is micrometers.")
        real_length_microns = real_length
    
    pixels_per_micron = pixel_length / real_length_microns
    
    return Scale(pixels_per_micron=pixels_per_micron, unit="µm")


def scale_detector(image: np.ndarray, visual_debug: bool = False) -> Dict:
    ruler_color = (0, 0, 255) # BGR
    ruler_color_range = create_color_range_from_bgr(ruler_color, tolerance=10)

    ruler = detect_scale_ruller(image, ruler_color_range)
    print(ruler)
    return ruler_data_to_scale(ruler)

