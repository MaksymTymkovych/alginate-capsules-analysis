import math

import numpy as np
from encapsu_view.entities.contour import Contour
from encapsu_view.entities.ellipse import Ellipse
from encapsu_view.entities.scale import Scale
from encapsu_view.entities.processed_capsule import ProcessedCapsule


def circle_contour(center, radius, num_points=720) -> Contour:
    angles = np.linspace(0, 2 * math.pi, num_points, endpoint=False)
    points = np.stack([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)], axis=1)
    return Contour(points=points.astype(np.float32), contour_type="inner_object")


# Synthetic capsule with known geometry: capsule radius 100 px, concentric core radius 50 px,
# scale 2 px/µm -> expected shell thickness 50 px = 25 µm, area ratio 0.25, form factor 1.0
ellipse = Ellipse(center=(150, 150), axes=(200, 200), angle=0)
inner_objects = [circle_contour(center=(150, 150), radius=50)]
scale = Scale(pixels_per_micron=2.0)

capsule = ProcessedCapsule(ellipse=ellipse, inner_objects=inner_objects, image_shape=(300, 300), capsule_id="synthetic_0")

# Save the processed capsule information to CSV: one row per experimental condition
capsule.save_to_csv("processed_capsule.csv", overwrite=True, field_date="17.01.2026", field_conditions="15%", scale=scale)
capsule.save_to_csv("processed_capsule.csv", overwrite=False, field_date="17.01.2026", field_conditions="20%", scale=scale)
capsule.save_to_csv("processed_capsule.csv", overwrite=False, field_date="17.01.2026", field_conditions="25%", scale=scale)
capsule.save_to_csv("processed_capsule.csv", overwrite=False, field_date="17.01.2026", field_conditions="30%", scale=scale)

row = capsule.to_row(scale)
for key in ['form_factor', 'ratio', 'core_circularity', 'capsule_radius_max_um', 'core_radius_max_um',
            'distance_to_core_min_um', 'distance_to_core_median_um', 'distance_to_core_max_um']:
    print(f"{key}: {row[key]:.3f}")

print(f"Processed capsule {capsule.id} saved to 'processed_capsule.csv'")
