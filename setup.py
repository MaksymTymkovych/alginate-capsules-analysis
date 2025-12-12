from setuptools import setup, find_packages
import subprocess
import sys

def check_tesseract():
    try:
        subprocess.run(["tesseract", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print(
            "\nWARNING: Tesseract OCR is not installed!\n"
            "Please install it to use pytesseract features.\n"
            "Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
            "Colab:      !apt-get install -y tesseract-ocr\n"
            "macOS:      brew install tesseract\n"
            "Windows:    https://github.com/tesseract-ocr/tesseract\n"
        )

check_tesseract()


setup(
    name="encapsu_view",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.23.1",
        "opencv-python-headless",
        "matplotlib==3.5.2",
        "matplotlib-inline==0.1.3",
        "anytree",
        "pytesseract",
        "pillow",
        "tqdm==4.64.0",
        "tabulate==0.8.10",
        "pandas==1.4.3",
        "torch==1.13.1",
        "torchvision==0.14.1",
        "streamlit==1.11.1",
        "huggingface_hub",
        "gradio",
    ],
    extras_require={
        "ocr": ["pytesseract"]
    },
    python_requires='>=3.8',
)
