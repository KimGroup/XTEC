from distutils.core import setup

setup(
    name="X-TEC",
    version="1.0.0",
    description="X-ray Temperature Clustering",
    url="https://cels.anl.gov/axmas/",
    packages=["xtec"],
    python_requires=">=3.7",
    install_requires=["numpy", "matplotlib", "scikit-learn", "scipy"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
