from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT="-e ."

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file:
        requirements=file.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        
    return requirements


setup(
    name="ML_project002",
    version="0.0.0.1",
    author="Nikhil Kumar Singh Parmar",
    author_email="nikhilkumarsingh5872@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)