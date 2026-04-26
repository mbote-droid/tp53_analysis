"""Setup configuration for TP53 Analysis Pipeline."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tp53-analysis",
    version="1.0.0",
    author="Samuel Mbote",
    author_email="mbotesamuel9@gmail.com",
    description="Comprehensive bioinformatics pipeline for TP53 gene analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tp53_analysis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "biopython>=1.79",
        "matplotlib>=3.3.0",
        "numpy>=1.19.0",
        "urllib3>=1.26.0",
    ],
    entry_points={
        "console_scripts": [
            "tp53-analysis=tp53_analysis:main",
        ],
    },
)