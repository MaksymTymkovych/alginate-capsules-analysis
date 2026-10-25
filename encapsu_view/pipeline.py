from dataclasses import dataclass
from typing import Optional, Union, BinaryIO
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

from encapsu_view.analysis.detected_data import DetectedData
from encapsu_view.analysis.processed_data import ProcessedData
from encapsu_view.analysis.detection import process_image_with_capsules, capsule_detector, inner_detector
from encapsu_view.analysis.scale import scale_detector
from encapsu_view.entities.scale import Scale
from encapsu_view.visualization.hierarchy import format_tree, plot_capsule_hierarchy


ImageSource = Union[str, os.PathLike, bytes, BinaryIO]

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_image(source: ImageSource) -> np.ndarray:
    """Read an image from a path, raw bytes or a file object and return it as 8-bit BGR."""
    if isinstance(source, (str, os.PathLike)):
        # np.fromfile + imdecode instead of cv2.imread: imread fails on non-ASCII paths on Windows
        data = np.fromfile(os.fspath(source), dtype=np.uint8)
    elif isinstance(source, (bytes, bytearray)):
        data = np.frombuffer(source, dtype=np.uint8)
    else:
        data = np.frombuffer(source.read(), dtype=np.uint8)

    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {source if isinstance(source, (str, os.PathLike)) else '<buffer>'}")

    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if len(img.shape) == 2 or img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 3:
        return img

    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    raise ValueError(f"Unsupported shape: {img.shape}")


@dataclass
class AnalysisResult:
    image: np.ndarray  # BGR
    detected: DetectedData
    processed: ProcessedData
    source: Optional[str] = None

    @property
    def scale(self) -> Optional[Scale]:
        return self.processed.scale

    def raw_visualization(self, rgb: bool = True) -> np.ndarray:
        vis = self.detected.visualize_detection(self.image)
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB) if rgb else vis

    def processed_visualization(self, rgb: bool = True) -> np.ndarray:
        vis = self.processed.visualize_detection(self.image)
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB) if rgb else vis

    def tree_text(self, unicode: bool = True) -> str:
        return format_tree(self.processed, unicode=unicode)

    def plot_hierarchy(self, show: bool = False) -> plt.Figure:
        return plot_capsule_hierarchy(self.processed, show=show)

    def to_dict(self) -> dict:
        data = self.processed.to_dict()
        if self.source is not None:
            data['source'] = self.source
        return data


def analyze_image(image: np.ndarray,
                  ellipse_method: str = "direct",
                  visual_debug: bool = False,
                  source: Optional[str] = None) -> AnalysisResult:
    """Run detection, scale recognition and ellipse fitting on a BGR image."""
    detected = process_image_with_capsules(image, capsule_detector, inner_detector, scale_detector, visual_debug)
    processed = detected.convert_to_processed_data(ellipse_method=ellipse_method, visual_debug=visual_debug)
    return AnalysisResult(image=image, detected=detected, processed=processed, source=source)


def analyze_file(path: Union[str, os.PathLike], **kwargs) -> AnalysisResult:
    return analyze_image(load_image(path), source=os.path.basename(os.fspath(path)), **kwargs)
