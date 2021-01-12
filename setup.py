import os
from setuptools import find_packages
from setuptools import setup


def get_long_description() -> str:
    readme_filepath = os.path.join(os.path.dirname(__file__), "README.md")
    with open(readme_filepath) as f:
        return f.read()


setup(
    name="kurobako",
    version="0.1.9",
    description="A Python library to help implementing kurobako's solvers and problems",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Takeru Ohta",
    author_email="phjgt308@gmail.com",
    url="https://github.com/sile/kurobako-py",
    license="MIT",
    packages=find_packages(),
    install_requires=["lupa", "numpy"],
    extras_require={"checking": ["hacking", "mypy", "black"]},
)
