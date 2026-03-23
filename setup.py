from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Mlops-Project-5",
    version="0.1",
    author="Himanshu Borikar",
    packages=find_packages(),
    install_requires=requirements
) 