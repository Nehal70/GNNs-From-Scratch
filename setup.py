"""
Setup script for GNNs-From-Scratch Python package
Install with: pip install .
Development install: pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gnns-from-scratch",
    version="0.1.0",
    author="Nehal",
    author_email="",  # Add your email
    description="From-scratch implementations of GNNs, Neural Networks, and ML algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nehal70/GNNs-From-Scratch",
    packages=find_packages(),
    package_data={
        "": ["LICENSE", "README.md"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "black>=21.0"],
    },
    keywords="machine-learning neural-networks graph-neural-networks cuda deep-learning from-scratch",
    project_urls={
        "Bug Reports": "https://github.com/Nehal70/GNNs-From-Scratch/issues",
        "Source": "https://github.com/Nehal70/GNNs-From-Scratch",
    },
)

