import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from encapsu_view.analysis.detection import process_image_with_capsules, capsule_detector, inner_detector
from encapsu_view.analysis.scale import scale_detector

from encapsu_view.visualization.hierarchy import visualize_tree, plot_capsule_hierarchy, export_to_json

base_dir = os.path.dirname(os.path.dirname(__file__)) 
test_dir = os.path.join(base_dir, 'data', 'test_images')



def example(visual_debug: bool = False):
    # for filename in os.listdir(test_dir):
    #     if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.tif'):
    #         path = os.path.join(test_dir, filename)

    IMAGE_NAME = os.path.join(test_dir, "Snap-21062.tif")
    image = cv2.imread(IMAGE_NAME)

    detected_data = process_image_with_capsules(image, capsule_detector, inner_detector, scale_detector, visual_debug)
    if visual_debug:
        detected_data.plot_detection(image)

    processed_data = detected_data.convert_to_processed_data(ellipse_method="direct", visual_debug=visual_debug)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    raw_vis = detected_data.visualize_detection(image)
    raw_vis_rgb = cv2.cvtColor(raw_vis, cv2.COLOR_BGR2RGB)
    ax1.imshow(raw_vis_rgb)
    ax1.set_title('Raw Detection (Contours)')
    ax1.axis('off')

    processed_vis = processed_data.visualize_detection(image)
    processed_vis_rgb = cv2.cvtColor(processed_vis, cv2.COLOR_BGR2RGB)
    ax2.imshow(processed_vis_rgb)
    ax2.set_title('Processed Detection (Ellipses)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

    visualize_tree(processed_data)
    
    json_data = processed_data.to_dict()
    print(json_data)

    # json_data = export_to_json(processed_data, "capsules_analysis.json")
    
    
    plot_capsule_hierarchy(processed_data)


if __name__ == "__main__":
    example(False)