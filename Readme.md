# Alginate Capsule Analysis

This project provides tools for analyzing alginate capsules in images using computer vision and capsule detection algorithms. It includes a Python package, helper notebooks, and a demo notebook that can be run directly in Google Colab.

---

## Project Structure

```
├── encapsu_view/ # Python package with core functionality
├── LICENSE # Project license
├── notebooks/ # Jupyter/Colab notebooks
│ └── demo.ipynb # Demo notebook for Google Colab
├── README.md # Project description (this file)
```
---

## Setup

You can install the project dependencies directly from `packages.txt` (or `requirements.txt` if you have one).  
For example:
```
pip install -e .

```
From git:

```
pip install git+https://github.com/MaksymTymkovych/alginate-capsules-analysis
```
---

## Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MaksymTymkovych/alginate-capsules-analysis/blob/main/notebooks/demo.ipynb)


--- 

## Usage

After installing dependencies, you can import and use the package in your Python scripts or notebooks:

```
from encapsu_view.analysis.detection import process_image_with_capsules, capsule_detector, inner_detector
from encapsu_view.analysis.scale import scale_detector
from encapsu_view.visualization.hierarchy import visualize_tree, plot_capsule_hierarchy

```

---
## License

This project is licensed under the MIT License. See LICENSE for details.

---
## Authors:
- Add all participants

