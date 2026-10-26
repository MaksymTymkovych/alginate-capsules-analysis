# Alginate Capsule Analysis

This project provides tools for analyzing alginate capsules in images using computer vision and capsule detection algorithms. It includes a Python package, helper notebooks, and a demo notebook that can be run directly in Google Colab.

---
## TODO List

- [x] Export to CSV (Table)
  - [x] Feature list (form coef, capsule area, core area, core area relation to capsule area, distances - min, max, median, mean, px, microns)
  - [x] Update entities
  - [x] Create functionality to export entity
  - [x] Create script to test functionality
- [x] Web server update
  - [x] One image inference
  - [x] Multiple image inference (in archive)
- [ ] Computer Vision update
  - [ ] 
  - [ ]
 
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

The notebook installs the package, starts the web app and prints a public `*.gradio.live` link.
To test unreleased changes, set `BRANCH` in the install cell (e.g. `"develop"`).


--- 

## Usage

After installing dependencies, you can import and use the package in your Python scripts or notebooks:

```
from encapsu_view.analysis.detection import process_image_with_capsules, capsule_detector, inner_detector
from encapsu_view.analysis.scale import scale_detector
from encapsu_view.visualization.hierarchy import visualize_tree, plot_capsule_hierarchy

```

Single image:

```
from encapsu_view.pipeline import analyze_file

result = analyze_file("image.tif")
df = result.to_dataframe(field_date="26.10.2026", field_conditions="15%")
```

Batch processing (folder or ZIP archive):

```
python -m encapsu_view.batch path/to/images -o batch_output --date 26.10.2026 --conditions "15%"
python -m encapsu_view.batch images.zip -o batch_output --all --zip
```

Output: `capsules.csv` (one row per capsule, complete capsules only unless `--all`), `summary.csv` (one row per image, with errors), `results.json` and `visualizations/`.

A batch is treated as a time series: images are ordered by file name in natural order (`frame_2` before `frame_10`),
and every capsule gets a `track_id` that stays the same across images. A shift of the whole field (microscope stage)
is compensated using complete capsules and reported in `summary.csv` (`field_shift_x_px`, `field_shift_y_px`).
Track ids are drawn on the visualizations. Use `--no-track` for independent images.

Web app (single image and batch tabs):

```
python -m encapsu_view.app            # http://localhost:7860
python -m encapsu_view.app --share    # public gradio.live link
```

From Python / a notebook:

```
from encapsu_view.app import launch
launch(share=True)
```

---
## Models
- todo

---
## Datasets
- todo

---
## License

This project is licensed under the MIT License. See LICENSE for details.

---
## Authors:
- Add all participants

---
## Citations:
- TODO

