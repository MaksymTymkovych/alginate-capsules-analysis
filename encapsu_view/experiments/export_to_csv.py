import numpy as np
from encapsu_view.entities.contour import Contour
from encapsu_view.entities.ellipse import Ellipse
from encapsu_view.entities.processed_capsule import ProcessedCapsule

# Create an ellipse and contours (dummy data for illustration)
ellipse = Ellipse(center=(50, 50), axes=(30, 20), angle=45)
inner_objects = [Contour(points=np.zeros(shape=(0,0))), Contour(points=np.zeros(shape=(0,0)))]  # Example inner objects

# Create a ProcessedCapsule instance
capsule = ProcessedCapsule(ellipse=ellipse, inner_objects=inner_objects, image_shape=(100, 100))

# Save the processed capsule information to CSV
capsule.save_to_csv("processed_capsule.csv", overwrite=True, field_date="17.01.2026", field_conditions="15%")
capsule.save_to_csv("processed_capsule.csv", overwrite=False, field_date="17.01.2026", field_conditions="20%")
capsule.save_to_csv("processed_capsule.csv", overwrite=False, field_date="17.01.2026", field_conditions="25%")
capsule.save_to_csv("processed_capsule.csv", overwrite=False, field_date="17.01.2026", field_conditions="30%")

print(f"Processed capsule {capsule.id} saved to 'processed_capsule.csv'")