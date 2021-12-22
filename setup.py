from distutils.core import setup

setup(
    name="XTEC",
    version="0.1.0",
    description="X-ray Temperature Clustering",
    author="Jordan Venderley",
    maintainer="Krishnanand Mallayya",
    maintainer_email="krishnanandmallayya@gmail.com",
    url="https://cels.anl.gov/axmas/",
    packages=["xtec"],
    download_url="https://github.com/KimGroup/XTEC",
    python_requires=">=3.7",
    install_requires=["numpy", "matplotlib", "scikit-learn", "scipy"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
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
