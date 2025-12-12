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
    packages=find_packages(include=["encapsu_view", "encapsu_view.*"]),
    install_requires=[
        "numpy",
        "opencv-python-headless",
        "matplotlib",
        "matplotlib-inline",
        "anytree",
        "pytesseract",
        "pillow",
        "tqdm",
        "tabulate",
        "pandas",
        "torch",
        "torchvision",
        "streamlit",
        "huggingface_hub",
        "gradio",
    ],
    extras_require={
        "ocr": ["pytesseract"]
    },
    python_requires='>=3.8',
)
