import sys
from pathlib import Path

from setuptools import find_packages, setup

_deps = [
    "diffusers>=0.36.0",
    "packaging>=20.0",
    "transformers>=4.57.3",
]

version_range_max = max(sys.version_info[1], 10) + 1

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Programming Language :: Python :: 3",
]

for minor_version in range(8, version_range_max):
    classifiers.append(f"Programming Language :: Python :: 3.{minor_version}")

setup(
    name="auto_diffusers",
    version="2.0.38",
    description="diffusers with search engine",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    keywords="huggingface civitai diffusers model search deep learning diffusion pytorch stable diffusion",
    license="Apache 2.0 License",
    author="suzukimain",
    author_email="gt13579552@gmail.com",
    url="https://github.com/suzukimain/auto_diffusers",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["py.typed"]},
    python_requires=">=3.8.0",
    include_package_data=True,
    install_requires=list(_deps),
    classifiers=classifiers,
)
