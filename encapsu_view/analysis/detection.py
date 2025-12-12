from typing import List, Callable, Dict
import numpy as np
import cv2
import pickle
import os
from huggingface_hub import hf_hub_download

from encapsu_view.analysis.detected_data import DetectedData
from encapsu_view.entities.contour import Contour
from encapsu_view.analysis.capsule_assembly import CapsuleAssembly

def process_image_with_capsules(image: np.ndarray,
                               capsule_detector: Callable[[np.ndarray], List[Contour]],
                               inner_object_detector: Callable[[np.ndarray], List[Contour]],
                               scale_detector: Callable[[np.ndarray], Dict],
                               visual_debug: bool = False) -> DetectedData:

    """
    ruler_color = (0, 0, 255) # BGR
    ruler_color_range = create_color_range_from_bgr(ruler_color, tolerance=10)
    ruler = detect_scale_ruller(image, ruler_color_range)

    scale = None
    if ruler:
        scale = ruler_data_to_scale(ruler)    
    """
    scale_data = scale_detector(image, visual_debug)
    #scale = ruler_data_to_scale(ruler)

    assembler = CapsuleAssembly(capsule_detector, inner_object_detector)
    capsules = assembler.process_image(image)

    return DetectedData(capsules, scale_data, image.shape[:2])


_capsule_predictor = None
_inner_predictor = None


def load_model_file(local_path: str, hf_repo: str, hf_filename: str):

    use_hf = os.getenv("USE_HF_MODELS", "0") == "1"

    if not use_hf and os.path.exists(local_path):
        return local_path

    return hf_hub_download(
        repo_id=hf_repo,
        filename=hf_filename
    )


def _initialize_capsule_predictor(predictor_path: str = "/media/maxim/big-500/datasets/microscopy/IMP-microscopy/dataset-1-it/IS_cfg.pickle", model_output_path: str = "/media/maxim/big-500/datasets/microscopy/IMP-microscopy/dataset-1-it/output/instance_segmentation/"):
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    global _capsule_predictor
    if _capsule_predictor is not None:
        return

    cfg_path = load_model_file(
        local_path=predictor_path,
        hf_repo="MaksymTymkovych/Alginate-Capsule-Detector",
        hf_filename="capsule-detector/IS_cfg.pickle"
    )

    model_path = load_model_file(
        local_path=os.path.join(model_output_path, "model_final.pth"),
        hf_repo="MaksymTymkovych/Alginate-Capsule-Detector",
        hf_filename="capsule-detector/model_final.pth"
    )
        
    with open(cfg_path, "rb") as f:
        cfg = pickle.load(f)
    
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    _capsule_predictor = DefaultPredictor(cfg)


def _initialize_inner_predictor(predictor_path: str = "/media/maxim/big-500/datasets/microscopy/IMP-microscopy/dataset-1-it/IS_cfg2.pickle", model_output_path: str = "/media/maxim/big-500/datasets/microscopy/IMP-microscopy/dataset-1-it/output2/instance_segmentation/"):
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    global _inner_predictor
    if _inner_predictor is not None:
        return
        
    cfg_path = load_model_file(
        local_path=predictor_path,
        hf_repo="MaksymTymkovych/Alginate-Capsule-Detector",
        hf_filename="inner-detector/IS_cfg2.pickle"
    )

    model_path = load_model_file(
        local_path=os.path.join(model_output_path, "model_final.pth"),
        hf_repo="MaksymTymkovych/Alginate-Capsule-Detector",
        hf_filename="inner-detector/model_final.pth"
    )
        
    with open(cfg_path, "rb") as f:
        cfg = pickle.load(f)
    
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    _inner_predictor = DefaultPredictor(cfg)

def capsule_detector(image: np.ndarray) -> List[Contour]:
    global _capsule_predictor
    
    if _capsule_predictor is None:
        _initialize_capsule_predictor()

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    outputs = _capsule_predictor(image)

    instances = outputs["instances"]
    masks = instances.pred_masks.cpu().numpy() if instances.has("pred_masks") else None
    classes = instances.pred_classes.cpu().numpy() if instances.has("pred_classes") else None
    scores = instances.scores.cpu().numpy() if instances.has("scores") else None
    
    contours = []
    
    if masks is not None and len(masks) > 0:
        for i, mask in enumerate(masks):

            mask_uint8 = (mask * 255).astype(np.uint8)
            

            contour_points, _ = cv2.findContours(
                mask_uint8, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for j, contour in enumerate(contour_points):
                if len(contour) >= 3:

                    epsilon = 0.001 * cv2.arcLength(contour, True)
                    approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # points = approx_contour.reshape(-1, 2)
                    points = contour.reshape(-1, 2)
                    
                    contour_type = "unknown"
                    if classes is not None and i < len(classes):
                        class_id = classes[i]
                        if class_id == 0:
                            contour_type = "capsule"
                        else:
                            contour_type = "unknown"
                    
                    contour_id = f"obj_{i}_contour_{j}"
                    if scores is not None and i < len(scores):
                        contour_id += f"_score_{scores[i]:.3f}"
                    
                    contour_obj = Contour(
                        points=points,
                        contour_id=contour_id,
                        contour_type=contour_type
                    )
                    
                    contours.append(contour_obj)
    
    return contours



def inner_detector(image: np.ndarray) -> List[Contour]:
    global _inner_predictor
    
    if _inner_predictor is None:
        _initialize_inner_predictor()

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    outputs = _inner_predictor(image)

    instances = outputs["instances"]
    masks = instances.pred_masks.cpu().numpy() if instances.has("pred_masks") else None
    classes = instances.pred_classes.cpu().numpy() if instances.has("pred_classes") else None
    scores = instances.scores.cpu().numpy() if instances.has("scores") else None
    
    contours = []
    
    if masks is not None and len(masks) > 0:
        for i, mask in enumerate(masks):

            mask_uint8 = (mask * 255).astype(np.uint8)
            

            contour_points, _ = cv2.findContours(
                mask_uint8, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for j, contour in enumerate(contour_points):
                if len(contour) >= 3:

                    epsilon = 0.001 * cv2.arcLength(contour, True)
                    approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # points = approx_contour.reshape(-1, 2)
                    points = contour.reshape(-1, 2)
                    
                    contour_type = "unknown"
                    if classes is not None and i < len(classes):
                        class_id = classes[i]
                        if class_id == 0:
                            contour_type = "inner_object"
                        else:
                            contour_type = "unknown"
                    
                    contour_id = f"obj_{i}_inner_object_{j}"
                    if scores is not None and i < len(scores):
                        contour_id += f"_score_{scores[i]:.3f}"
                    
                    contour_obj = Contour(
                        points=points,
                        contour_id=contour_id,
                        contour_type=contour_type
                    )
                    
                    contours.append(contour_obj)
    
    return contours
