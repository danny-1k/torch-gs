import setuptools

long_description = open("README.md", "r").read()
requirements = [i.strip() for i in open("requirements.txt", "r").readlines()]

setuptools.setup(
    name="torchgs",
    version="0.0.1",
    author="Daniel Ik",
    author_email="codingeinstein@gmail.com",
    url="https://github.com/danny-1k/torch-gs",
    keywords=[
        "pytorch", "machine-learning", "deep-learning",
        "deep learning", "machine learning", "grid-search",
        "grid search"
    ],
    description="Pytorch wrapper for performing grid-search",
    long_description_content_type='text/markdown',
    long_description=long_description,
    packages=setuptools.find_packages(),
    classifiers=[
        'Intended Audience :: Developers',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    py_modules=["torchgs"],
    package_dir={"": "."},
    install_requires=requirements
)
