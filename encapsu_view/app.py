from typing import List, Optional
import argparse
import os
import shutil
import tempfile
import zipfile

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt

from encapsu_view.pipeline import analyze_file, SUPPORTED_EXTENSIONS
from encapsu_view.batch import analyze_files, find_images, safe_extract, strip_contour_points
from encapsu_view.entities.processed_capsule import write_csv_rows

# The app renders figures to images, it never opens GUI windows
matplotlib.use("Agg")

IMAGE_FILE_TYPES = list(SUPPORTED_EXTENSIONS)
BATCH_FILE_TYPES = IMAGE_FILE_TYPES + [".zip"]


def _new_work_dir() -> str:
    return tempfile.mkdtemp(prefix="encapsu_app_")


def run_single(file_path: Optional[str], field_date: str, field_conditions: str, include_cropped: bool):
    if not file_path:
        raise gr.Error("Upload an image first")

    try:
        result = analyze_file(file_path)
    except Exception as e:
        raise gr.Error(f"Analysis failed: {type(e).__name__}: {e}")

    df = result.to_dataframe(field_date, field_conditions, only_complete=not include_cropped)

    csv_path = None
    if not df.empty:
        name = os.path.splitext(os.path.basename(file_path))[0]
        csv_path = os.path.join(_new_work_dir(), f"{name}_capsules.csv")
        write_csv_rows(csv_path, df.to_dict('records'), overwrite=True)

    fig = result.plot_hierarchy(show=False)
    # Gradio renders the figure itself; closing only detaches it from pyplot so figures do not pile up
    plt.close(fig)

    scale_note = (f"Scale: {result.scale.pixels_per_micron:.3f} px/µm" if result.scale
                  else "Scale ruler not found: measurements are in pixels only")

    return (
        result.raw_visualization(),
        result.processed_visualization(),
        scale_note,
        df,
        csv_path,
        result.tree_text(unicode=True),
        fig,
        strip_contour_points(result.to_dict()),
    )


def _collect_batch_inputs(file_paths: List[str], inputs_dir: str) -> List[str]:
    """Copy uploaded images and extract uploaded ZIPs into one folder, so names inside the batch stay readable."""
    for path in file_paths:
        name = os.path.basename(path)
        if name.lower().endswith(".zip"):
            target = os.path.join(inputs_dir, os.path.splitext(name)[0])
            with zipfile.ZipFile(path, 'r') as zip_file:
                safe_extract(zip_file, target)
        elif name.lower().endswith(SUPPORTED_EXTENSIONS):
            shutil.copy(path, os.path.join(inputs_dir, name))
    return find_images(inputs_dir)


def run_batch(files, field_date: str, field_conditions: str, include_cropped: bool, track: bool,
              progress=gr.Progress(track_tqdm=True)):
    if not files:
        raise gr.Error("Upload a ZIP archive or images first")

    # Depending on the Gradio version, uploaded files come as paths or as objects with .name
    file_paths = [f if isinstance(f, str) else f.name for f in (files if isinstance(files, list) else [files])]

    work_dir = _new_work_dir()
    inputs_dir = os.path.join(work_dir, "inputs")
    output_dir = os.path.join(work_dir, "results")
    os.makedirs(inputs_dir)

    image_paths = _collect_batch_inputs(file_paths, inputs_dir)
    if not image_paths:
        raise gr.Error(f"No images found. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")

    batch = analyze_files(image_paths, output_dir=output_dir, base_dir=inputs_dir,
                          field_date=field_date, field_conditions=field_conditions, track=track)

    only_complete = not include_cropped
    zip_path = batch.to_zip(os.path.join(work_dir, "capsule_analysis_results.zip"), only_complete=only_complete)

    gallery = [(item.visualization_path, item.source) for item in batch.items if item.visualization_path]
    status = f"Processed {len(batch.items)} images, {len(batch.failed)} failed"

    return gallery, status, batch.summary_dataframe(), batch.to_dataframe(only_complete), zip_path


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Alginate Capsule Analysis") as demo:
        gr.Markdown("# Alginate Capsule Analysis")

        with gr.Tabs():
            with gr.Tab("Single image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        single_file = gr.File(label="Image", file_types=IMAGE_FILE_TYPES, type="filepath")
                        single_date = gr.Textbox(label="Date", placeholder="e.g. 26.10.2026")
                        single_conditions = gr.Textbox(label="Conditions", placeholder="e.g. 15%")
                        single_cropped = gr.Checkbox(label="Include cropped capsules in the table", value=False)
                        single_button = gr.Button("Run analysis", variant="primary")
                    with gr.Column(scale=2):
                        with gr.Row():
                            raw_image = gr.Image(label="Detection (contours)", type="numpy")
                            processed_image = gr.Image(label="Processed (ellipses)", type="numpy")
                        scale_info = gr.Markdown()

                capsules_table = gr.Dataframe(label="Capsule measurements", wrap=True)
                single_csv = gr.File(label="Download CSV")
                with gr.Accordion("Details", open=False):
                    tree_text = gr.Textbox(label="Capsule tree", lines=20)
                    hierarchy_plot = gr.Plot(label="Hierarchy plot")
                    json_output = gr.JSON(label="Full result (JSON)")

                single_button.click(
                    run_single,
                    inputs=[single_file, single_date, single_conditions, single_cropped],
                    outputs=[raw_image, processed_image, scale_info, capsules_table, single_csv,
                             tree_text, hierarchy_plot, json_output],
                )

            with gr.Tab("Batch"):
                gr.Markdown("Upload a ZIP archive with images, or several images at once. "
                            "Images are processed in natural file name order (frame_2 before frame_10).")
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_files = gr.File(label="ZIP archive or images", file_types=BATCH_FILE_TYPES,
                                              file_count="multiple", type="filepath")
                        batch_date = gr.Textbox(label="Date", placeholder="e.g. 26.10.2026")
                        batch_conditions = gr.Textbox(label="Conditions", placeholder="e.g. 15%")
                        batch_cropped = gr.Checkbox(label="Include cropped capsules in the table", value=False)
                        batch_track = gr.Checkbox(
                            label="Time series: track capsules across images (track_id, stage shift compensation)",
                            value=True)
                        batch_button = gr.Button("Run batch analysis", variant="primary")
                    with gr.Column(scale=2):
                        batch_status = gr.Markdown()
                        batch_zip = gr.File(label="Download results (CSV, JSON, visualizations)")
                        gallery = gr.Gallery(label="Processed images", columns=4)

                summary_table = gr.Dataframe(label="Summary per image", wrap=True)
                batch_table = gr.Dataframe(label="Capsule measurements", wrap=True)

                batch_button.click(
                    run_batch,
                    inputs=[batch_files, batch_date, batch_conditions, batch_cropped, batch_track],
                    outputs=[gallery, batch_status, summary_table, batch_table, batch_zip],
                )

    return demo


def launch(share: bool = False, server_name: Optional[str] = None, server_port: Optional[int] = None, **kwargs):
    """Start the web app. In Colab use share=True to get a public link."""
    demo = build_demo()
    demo.queue()
    demo.launch(share=share, server_name=server_name, server_port=server_port, **kwargs)
    return demo


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Alginate capsule analysis web app")
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    parser.add_argument("--host", default=None, help="server host, e.g. 0.0.0.0 to allow access from the network")
    parser.add_argument("--port", type=int, default=None, help="server port (default: 7860)")
    args = parser.parse_args(argv)

    launch(share=args.share, server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
