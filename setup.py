from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="spotify-yapping-detector",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning system that classifies Spotify podcasts as 'yapping' vs 'normal' content",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/spotify-yapping-detector",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "jupyter>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "spotify-processor=scripts.spotify_processor:main",
            "yapping-train=scripts.train:main",
        ],
    },
)
