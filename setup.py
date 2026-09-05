from setuptools import setup, find_packages

setup(
    name="nutrigraph",
    version="0.1.0",
    description="NutriGraph: a knowledge graph linking foods to human molecular targets",
    author="Abhinav Sikhwal",
    author_email="abhisikhwal@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "pyyaml",
        "requests",
        "scikit-learn>=1.3",
        "scipy>=1.11",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "jupyter",
        ],
        "chem": [
            "rdkit",
        ],
        "bio": [
            "biopython",
            "bioservices",
        ],
        "geo": [
            "geopandas",
            "rasterio",
            "xarray",
        ],
        "ml": [
            "torch",
            "pytorch-lightning",
            "transformers",
        ],
    },
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
