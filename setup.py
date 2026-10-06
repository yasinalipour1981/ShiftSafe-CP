from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
requirements = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
]

setup(
    name="shiftsafe-cp",
    version="1.0.2",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.11",
    description=(
        "Cross-laboratory calibrated prediction intervals "
        "for concrete compressive strength"
    ),
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)
