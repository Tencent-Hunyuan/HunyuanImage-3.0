from setuptools import setup, find_packages
import os

# Read README.md for long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Hunyuan Image Generation Model 3.0"

# Parse requirements from requirements.txt
def parse_requirements(filename):
    requirements = []
    try:
        with open(filename, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
    return requirements

setup(
    name="hunyuan-image-3",
    version="3.0.0",
    author="Tencent Hunyuan",
    author_email="hunyuan@tencent.com",
    description="Hunyuan Image Generation Model 3.0 - Advanced AI Image Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tencent-Hunyuan/HunyuanImage-3.0",
    
    # Package configuration
    packages=find_packages(include=["hunyuan_image_3", "hunyuan_image_3.*"]),
    
    # Python version requirement
    python_requires=">=3.12",
    
    # Dependencies
    install_requires=parse_requirements("requirements.txt"),
    
    # Include package data files
    include_package_data=True,
    package_data={
        "hunyuan_image_3": [
            "assets/**/*",
            "PE/**/*",
            "*.py",
            "*.yaml",
            "*.json",
            "*.txt"
        ],
    },
    
    # Package classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
    
    # Keywords for package discovery
    keywords="ai, image-generation, computer-vision, deep-learning, hunyuan",
    
    # Project URLs
    project_urls={
        "Documentation": "https://github.com/Tencent-Hunyuan/HunyuanImage-3.0",
        "Source Code": "https://github.com/Tencent-Hunyuan/HunyuanImage-3.0",
        "Bug Reports": "https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/issues",
    },
    
    # Console entry points
    entry_points={
        "console_scripts": [
            "hunyuan-image=run_image_gen:main",
        ],
    },
    
    # Additional metadata
    license="TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT",
    license_files=("LICENSE",),
    platforms=["any"],
)