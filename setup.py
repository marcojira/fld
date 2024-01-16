with open("requirements.txt") as f:
    requirements = f.read().splitlines()

from setuptools import setup, find_packages

print(find_packages())
setup(
    name="fld",
    version="1.0.0",
    packages=find_packages(),
    install_requires=requirements,
)
