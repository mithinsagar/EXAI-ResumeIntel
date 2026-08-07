"""
EXAI-ResumeIntel: Setup configuration
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
"""

from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8")
REQUIREMENTS = [
    line.strip()
    for line in (HERE / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="exai-resumeintel",
    version="1.0.0",
    author="Mithin Sagar S",
    author_email="mithinsagar@gmail.com",
    description=(
        "An Explainable AI Framework for Automated Resume Analysis "
        "Using Shapley Values, LIME, and Domain Skill Ontology"
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/mithinsagar/EXAI-ResumeIntel",
    project_urls={
        "Bug Tracker": "https://github.com/mithinsagar/EXAI-ResumeIntel/issues",
        "Documentation": "https://github.com/mithinsagar/EXAI-ResumeIntel/tree/main/docs",
        "Source Code": "https://github.com/mithinsagar/EXAI-ResumeIntel",
    },
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "notebooks.*"]),
    python_requires=">=3.10",
    install_requires=REQUIREMENTS,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords=[
        "explainable-ai",
        "resume-analysis",
        "shapley-values",
        "lime",
        "tf-idf",
        "latent-semantic-analysis",
        "skill-ontology",
        "natural-language-processing",
    ],
    entry_points={
        "console_scripts": [
            "exai-serve=api.server:main",
            "exai-train=training.train:main",
            "exai-evaluate=training.evaluate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
