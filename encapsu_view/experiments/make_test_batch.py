import os
import shutil
import zipfile

import cv2
import numpy as np

base_dir = os.path.dirname(os.path.dirname(__file__))
source_image = os.path.join(base_dir, 'data', 'test_images', 'Snap-21062.tif')
batch_dir = os.path.join(base_dir, 'data', 'test_batch')
batch_zip = os.path.join(base_dir, 'data', 'test_batch.zip')
copies_dir = os.path.join(base_dir, 'data', 'test_batch_copies')
copies_zip = os.path.join(base_dir, 'data', 'test_batch_copies.zip')


def save(path: str, image: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imencode(os.path.splitext(path)[1], image)[1].tofile(path)


def make_test_batch():
    """Build a test batch from copies of one image, covering the cases the batch must handle."""
    if os.path.exists(batch_dir):
        shutil.rmtree(batch_dir)

    os.makedirs(batch_dir)
    image = cv2.imread(source_image)

    # 1. Unchanged copy
    shutil.copy(source_image, os.path.join(batch_dir, 'original.tif'))

    # 2. Bottom with the scale ruler cut off -> no scale, measurements only in px
    save(os.path.join(batch_dir, 'no_ruler.png'), image[:1500])

    # 3. 16-bit grayscale TIFF
    gray16 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint16) * 256
    save(os.path.join(batch_dir, 'gray16.tif'), gray16)

    # 4. Half resolution JPEG -> pixels_per_micron should be about half of the original
    save(os.path.join(batch_dir, 'half_size.jpg'), cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))

    # 5. Nested folder, mirrored image -> ruler text is mirrored
    save(os.path.join(batch_dir, 'series_2', 'mirrored.png'), cv2.flip(image, 1))

    # 6. Corrupted file with an image extension -> must be reported as an error, not stop the batch
    with open(os.path.join(batch_dir, 'broken.png'), 'wb') as f:
        f.write(b'not an image')

    # 7. Non-image file -> must be skipped
    with open(os.path.join(batch_dir, 'notes.txt'), 'w') as f:
        f.write('not an image either')

    pack_zip(batch_dir, batch_zip)

    print(f"Test batch: {batch_dir}")
    print(f"Test zip:   {batch_zip}")


def make_copies_batch(count: int = 10):
    """Build a batch of identical copies with the scale ruler: every image must give the same result."""
    if os.path.exists(copies_dir):
        shutil.rmtree(copies_dir)
    os.makedirs(copies_dir)

    for i in range(1, count + 1):
        shutil.copy(source_image, os.path.join(copies_dir, f'copy_{i:02d}.tif'))

    pack_zip(copies_dir, copies_zip)

    print(f"Copies batch: {copies_dir} ({count} images)")
    print(f"Copies zip:   {copies_zip}")


def pack_zip(folder: str, zip_path: str):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder):
            for name in files:
                path = os.path.join(root, name)
                zipf.write(path, arcname=os.path.relpath(path, folder))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create test batches from Snap-21062.tif")
    parser.add_argument("--copies", type=int, metavar="N",
                        help="create N identical copies with the scale ruler instead of the edge-case batch")
    args = parser.parse_args()

    if args.copies:
        make_copies_batch(args.copies)
    else:
        make_test_batch()
