from setuptools import setup
from setuptools import find_packages

extras = {}

setup(
    name="lazy_init",
    version="0.0.1",
    description="Lazy Init",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning",
    license="Apache",
    author="Sylvain Gugger",
    author_email="sylvain@huggingface.co",
    url="https://github.com/sgugger/lazy_init",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.6.0",
    install_requires=[],
    extras_require=extras,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
