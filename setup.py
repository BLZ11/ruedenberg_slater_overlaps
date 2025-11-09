"""
Setup script for ruedenberg_slater_overlaps package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ruedenberg_slater_overlaps",
    version="1.0.0",
    author="BLZ11",
    description="Python implementation of Ruedenberg's analytical expressions for Slater-type orbital overlaps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BLZ11/ruedenberg_slater_overlaps",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "matplotlib>=3.0",
            "jupyter>=1.0",
        ],
    },
    keywords="quantum chemistry, slater orbitals, overlap integrals, ruedenberg, roothaan",
)
