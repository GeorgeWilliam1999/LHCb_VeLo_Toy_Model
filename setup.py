"""
Setup file for the velo_toy package.

Install with: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="velo_toy",
    version="0.1.0",
    description="Quantum-inspired Hamiltonian track reconstruction for LHCb VELO",
    author="George Scriven",
    author_email="",
    url="",
    
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    python_requires=">=3.8",
    
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "pandas>=1.3",
        "matplotlib>=3.4",
        "dill>=0.3",
        "tqdm>=4.60",
    ],
    
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "isort>=5.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "velo-run-batch=bin.run_batch:main",
            "velo-aggregate=bin.aggregate_results:main",
            "velo-submit=bin.submit_experiment:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
