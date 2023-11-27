with open("requirements.txt") as f:
    requirements = f.read().splitlines()

from setuptools import setup, find_packages

print(find_packages())
setup(
    name="fls",
    version="0.1.2",
    packages=find_packages(),
    install_requires=requirements,
)
