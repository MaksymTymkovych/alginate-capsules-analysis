from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Tuple
import argparse
import json
import os
import shutil
import tempfile
import zipfile

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from encapsu_view.pipeline import analyze_file, SUPPORTED_EXTENSIONS
from encapsu_view.entities.processed_capsule import write_csv_rows
from encapsu_view.tracking import TrackerConfig, track_capsules, natural_sort_key


PathLike = Union[str, os.PathLike]

CAPSULES_CSV = "capsules.csv"
SUMMARY_CSV = "summary.csv"
RESULTS_JSON = "results.json"
VISUALIZATIONS_DIR = "visualizations"


@dataclass
class BatchItem:
    source: str  # path relative to the batch input, used as image_name
    rows: List[dict] = field(default_factory=list)  # all capsules, including cropped ones
    data: Optional[dict] = None
    visualization_path: Optional[str] = None
    error: Optional[str] = None
    # Set by tracking: position in the time series and field (stage) shift relative to the first frame, px
    frame: Optional[int] = None
    field_shift: Optional[Tuple[float, float]] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def summary(self) -> dict:
        statistics = (self.data or {}).get('statistics', {})
        scale = (self.data or {}).get('scale')
        summary = {}
        if self.frame is not None:
            summary['frame'] = self.frame
        summary.update({
            'image_name': self.source,
            'status': 'ok' if self.ok else 'error',
            'error': self.error or "",
            'total_capsules': statistics.get('total_capsules', 0),
            'complete_capsules': statistics.get('complete_capsules', 0),
            'cropped_capsules': statistics.get('cropped_capsules', 0),
            'total_inner_objects': statistics.get('total_inner_objects', 0),
            'pixels_per_micron': scale['pixels_per_micron'] if scale else None,
        })
        if self.field_shift is not None:
            summary['field_shift_x_px'], summary['field_shift_y_px'] = self.field_shift
        return summary


@dataclass
class BatchResult:
    items: List[BatchItem]
    output_dir: Optional[str] = None

    @property
    def failed(self) -> List[BatchItem]:
        return [item for item in self.items if not item.ok]

    def capsule_rows(self, only_complete: bool = True) -> List[dict]:
        return [row for item in self.items for row in item.rows
                if row['is_complete'] or not only_complete]

    def to_dataframe(self, only_complete: bool = True) -> pd.DataFrame:
        return pd.DataFrame(self.capsule_rows(only_complete))

    def summary_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([item.summary() for item in self.items])

    def to_dict(self) -> dict:
        return {
            'images': {item.source: item.data if item.ok else {'error': item.error} for item in self.items},
            'summary': [item.summary() for item in self.items],
        }

    def save(self, output_dir: Optional[PathLike] = None, only_complete: bool = True) -> Dict[str, str]:
        """Write capsules.csv, summary.csv and results.json; returns their paths."""
        output_dir = os.path.normpath(os.fspath(output_dir or self.output_dir or "."))
        os.makedirs(output_dir, exist_ok=True)

        paths = {
            'capsules': os.path.join(output_dir, CAPSULES_CSV),
            'summary': os.path.join(output_dir, SUMMARY_CSV),
            'json': os.path.join(output_dir, RESULTS_JSON),
        }

        rows = self.capsule_rows(only_complete)
        if rows:
            write_csv_rows(paths['capsules'], rows, overwrite=True)
        else:
            # Keep the file so that downstream tools always find it
            open(paths['capsules'], 'w', encoding='utf-8-sig').close()

        write_csv_rows(paths['summary'], [item.summary() for item in self.items], overwrite=True)

        with open(paths['json'], 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=_json_default)

        return paths

    def to_zip(self, zip_path: PathLike, only_complete: bool = True) -> str:
        """Save results (if not saved yet) and pack the output directory with visualizations into a ZIP."""
        if self.output_dir is None:
            raise ValueError("BatchResult has no output_dir: run the batch with output_dir to build a ZIP")

        self.save(self.output_dir, only_complete=only_complete)
        zip_path = os.fspath(zip_path)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(self.output_dir):
                for name in files:
                    path = os.path.join(root, name)
                    if os.path.abspath(path) == os.path.abspath(zip_path):
                        continue
                    zipf.write(path, arcname=os.path.relpath(path, self.output_dir))
        return zip_path


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def strip_contour_points(data: dict) -> dict:
    # Contour points make results.json huge; summary values are kept
    for capsule in data.get('capsules', []):
        for inner in capsule.get('inner_objects', []):
            inner.pop('points', None)
    return data


def find_images(folder: PathLike, recursive: bool = True) -> List[str]:
    """Images in natural order (frame_2 before frame_10): the order defines the time series."""
    folder = os.fspath(folder)
    found = []
    for root, dirs, files in os.walk(folder):
        # Skip macOS archive junk and hidden folders
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__MACOSX']
        for name in files:
            if name.startswith('.') or not name.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
            found.append(os.path.join(root, name))
        if not recursive:
            break
    return sorted(found, key=lambda path: natural_sort_key(os.path.relpath(path, folder)))


def _capsule_radius(row: dict) -> float:
    return (row['capsule_radius_min_px'] + row['capsule_radius_max_px']) / 2


def _with_tracking_columns(row: dict, frame: int, track_id: int) -> dict:
    # 'frame' goes before image_name, 'track_id' right after capsule_id, so they are easy to find in the table
    out = {}
    for key, value in row.items():
        if key == 'image_name':
            out['frame'] = frame
        out[key] = value
        if key == 'capsule_id':
            out['track_id'] = track_id
    return out


def _draw_track_ids(item: BatchItem):
    image = cv2.imdecode(np.fromfile(item.visualization_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    height, width = image.shape[:2]
    font_scale = max(1.0, width / 800)
    thickness = max(2, int(font_scale * 2))
    for row in item.rows:
        label = f"#{row['track_id']}"
        # Cropped capsules may have the ellipse center outside the image
        x = int(min(max(row['center_x_px'], 0), width - 1))
        y = int(min(max(row['center_y_px'], 0), height - 1))
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x = min(max(x - text_w // 2, 0), width - text_w)
        y = min(max(y + text_h // 2, text_h), height - 1)
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3)
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
    cv2.imencode(".png", image)[1].tofile(item.visualization_path)


def apply_tracking(items: List[BatchItem], config: Optional[TrackerConfig] = None):
    """Treat the items as an ordered time series and give every capsule a persistent track_id."""
    frames = [[(row['center_x_px'], row['center_y_px'], _capsule_radius(row), row['is_complete'])
               for row in item.rows]
              for item in items]
    result = track_capsules(frames, config)

    for frame, (item, track_ids, shift) in enumerate(zip(items, result.track_ids, result.field_shifts)):
        item.frame = frame
        item.field_shift = shift
        item.rows = [_with_tracking_columns(row, frame, track_id) for row, track_id in zip(item.rows, track_ids)]
        if item.visualization_path and item.rows:
            _draw_track_ids(item)


def analyze_files(paths: List[PathLike],
                  output_dir: Optional[PathLike] = None,
                  base_dir: Optional[PathLike] = None,
                  field_date: str = "",
                  field_conditions: str = "",
                  progress: bool = True,
                  track: bool = True,
                  tracker_config: Optional[TrackerConfig] = None,
                  **analyze_kwargs) -> BatchResult:
    """Analyze images one by one; a failure on one file is recorded and does not stop the batch.

    If output_dir is given, a visualization is saved for every image right away,
    so images do not have to be kept in memory.
    With track=True the paths are treated as a time series in the given order,
    and capsules get a persistent track_id across images.
    """
    if output_dir is not None:
        output_dir = os.fspath(output_dir)
        os.makedirs(os.path.join(output_dir, VISUALIZATIONS_DIR), exist_ok=True)

    items = []
    for path in tqdm(paths, desc="Analyzing", disable=not progress):
        path = os.fspath(path)
        source = os.path.relpath(path, base_dir).replace(os.sep, "/") if base_dir else os.path.basename(path)
        item = BatchItem(source=source)

        try:
            result = analyze_file(path, **analyze_kwargs)
            result.source = source
            item.rows = result.to_rows(field_date, field_conditions, only_complete=False)
            item.data = strip_contour_points(result.to_dict())

            if output_dir is not None:
                vis_name = os.path.splitext(source.replace("/", "__"))[0] + ".png"
                item.visualization_path = os.path.join(output_dir, VISUALIZATIONS_DIR, vis_name)
                vis = result.processed_visualization(rgb=False)
                # imencode + tofile instead of imwrite: imwrite fails on non-ASCII paths on Windows
                cv2.imencode(".png", vis)[1].tofile(item.visualization_path)
        except Exception as e:
            # Show the name inside the batch, not a temporary extraction path
            item.error = f"{type(e).__name__}: {e}".replace(path, source)
            print(f"Warning: failed to analyze {source}: {item.error}")

        items.append(item)

    if track:
        apply_tracking(items, tracker_config)

    return BatchResult(items=items, output_dir=output_dir)


def analyze_folder(folder: PathLike, recursive: bool = True, **kwargs) -> BatchResult:
    return analyze_files(find_images(folder, recursive), base_dir=folder, **kwargs)


def safe_extract(zip_file: zipfile.ZipFile, target_dir: str):
    # Protect against entries like "../../file" escaping the target directory
    target_dir = os.path.abspath(target_dir)
    for member in zip_file.namelist():
        member_path = os.path.abspath(os.path.join(target_dir, member))
        if os.path.commonpath([target_dir, member_path]) != target_dir:
            raise ValueError(f"Unsafe path in archive: {member}")
    zip_file.extractall(target_dir)


def analyze_zip(zip_path: PathLike, **kwargs) -> BatchResult:
    extract_dir = tempfile.mkdtemp(prefix="encapsu_batch_")
    try:
        with zipfile.ZipFile(os.fspath(zip_path), 'r') as zip_file:
            safe_extract(zip_file, extract_dir)
        return analyze_folder(extract_dir, **kwargs)
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Batch analysis of alginate capsule images")
    parser.add_argument("input", help="folder with images or a .zip archive")
    parser.add_argument("-o", "--output", default="batch_output", help="output folder (default: batch_output)")
    parser.add_argument("--date", default="", help="value for the 'date' column")
    parser.add_argument("--conditions", default="", help="value for the 'conditions' column")
    parser.add_argument("--all", action="store_true", help="include cropped capsules in capsules.csv")
    parser.add_argument("--zip", action="store_true", help="also pack results into <output>.zip")
    parser.add_argument("--no-track", action="store_true",
                        help="do not track capsules across images (images are independent, not a time series)")
    args = parser.parse_args(argv)

    run = analyze_zip if args.input.lower().endswith(".zip") else analyze_folder
    batch = run(args.input, output_dir=args.output, field_date=args.date, field_conditions=args.conditions,
                track=not args.no_track)

    paths = batch.save(only_complete=not args.all)
    if args.zip:
        paths['zip'] = batch.to_zip(os.path.normpath(args.output) + ".zip", only_complete=not args.all)

    print(f"\nProcessed {len(batch.items)} images, {len(batch.failed)} failed")
    for item in batch.failed:
        print(f"  {item.source}: {item.error}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
