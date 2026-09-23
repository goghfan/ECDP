from setuptools import setup, find_packages

setup(
    name="ecdp",
    version="1.0.0",
    description="Evidence-Calibrated Displacement Pyramids for Deformable Medical Image Registration",
    author="ECDP Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "nibabel>=3.2.0",
        "scipy>=1.7.0",
        "pyyaml>=5.4.0",
    ],
)
