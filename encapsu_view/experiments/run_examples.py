import os
import json

import matplotlib.pyplot as plt

from encapsu_view.pipeline import analyze_file
from encapsu_view.visualization.hierarchy import visualize_tree

base_dir = os.path.dirname(os.path.dirname(__file__))
test_dir = os.path.join(base_dir, 'data', 'test_images')



def example(visual_debug: bool = False):
    IMAGE_NAME = os.path.join(test_dir, "Snap-21062.tif")

    result = analyze_file(IMAGE_NAME, visual_debug=visual_debug)
    if visual_debug:
        result.detected.plot_detection(result.image)

    result.processed.save_to_csv(filename="csv.csv", overwrite=True, field_date="25.01.2026", field_conditions="15%", field_image_name="Snap-21062.tif")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.imshow(result.raw_visualization())
    ax1.set_title('Raw Detection (Contours)')
    ax1.axis('off')

    ax2.imshow(result.processed_visualization())
    ax2.set_title('Processed Detection (Ellipses)')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    visualize_tree(result.processed)

    json_data = result.to_dict()
    print(json.dumps(json_data['statistics'], indent=2))

    result.plot_hierarchy(show=True)


if __name__ == "__main__":
    example(False)
