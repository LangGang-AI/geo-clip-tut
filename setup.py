"""
GeoCLIP Package Setup Configuration
=================================

Installation and distribution configuration for GeoCLIP package.
Manages dependencies, metadata, and package structure for PyPI distribution.

Configuration Details:
- Package Name: geoclip
- Version: 1.2.0
- Python Requirements: >=3.6
- License: MIT
- Repository: github.com/VicenteVivan/geo-clip
"""

from setuptools import setup, find_packages

# Load requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    # Core package information
    name="geoclip",
    version="1.2.0",
    packages=find_packages(),
    
    # Package description and documentation
    description="",  # TODO: Add short description
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    
    # Project URLs and contact information
    url="https://github.com/VicenteVivan/geo-clip",
    author="Vicente Vivanco",
    author_email="vicente.vivancocepeda@ucf.edu",
    
    # Licensing and requirements
    license="MIT",
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.6",
    
    # Package classifiers for PyPI
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
