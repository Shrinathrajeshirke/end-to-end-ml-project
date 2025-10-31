from setuptools import find_packages, setup
from typing import List

e_dot = "-e ."

def get_requirements(file_path:str)->List[str]:
    '''
    this function return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        [req.replace("\n","") for req in requirements ]

        ## to remove -e. from requirements
        if e_dot in requirements:
            requirements.remove(e_dot)
    
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='shri',
    author_email='shrinathrajeshirke@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)