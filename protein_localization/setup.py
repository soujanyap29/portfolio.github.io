"""
Setup script for Protein Sub-Cellular Localization Pipeline
"""
from setuptools import setup, find_packages
import os

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read README
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='protein-localization-pipeline',
    version='1.0.0',
    author='Soujanya Patil',
    author_email='',
    description='Complete pipeline for protein sub-cellular localization in neurons',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/soujanyap29/portfolio.github.io',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'protein-localization=main:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.md', '*.ipynb'],
    },
    keywords='protein localization microscopy cellpose graph-neural-networks deep-learning',
    project_urls={
        'Bug Reports': 'https://github.com/soujanyap29/portfolio.github.io/issues',
        'Source': 'https://github.com/soujanyap29/portfolio.github.io',
    },
)
