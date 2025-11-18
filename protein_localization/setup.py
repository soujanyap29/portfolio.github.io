"""
Setup script for the Protein Localization Pipeline
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip() 
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith('#')
    ]

setup(
    name="protein_localization",
    version="1.0.0",
    author="Portfolio Project",
    description="Complete pipeline for protein sub-cellular localization analysis in neurons",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/soujanyap29/portfolio.github.io",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "gpu": [
            "torch-scatter",
            "torch-sparse",
            "torch-cluster",
        ],
    },
    entry_points={
        "console_scripts": [
            "protein-pipeline=src.preprocessing:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md"],
    },
    keywords=[
        "bioinformatics",
        "neuroscience",
        "protein-localization",
        "graph-neural-networks",
        "image-segmentation",
        "deep-learning",
        "cellpose",
    ],
)
