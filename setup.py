from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    """
    This function will return the list of requirements
    """
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if "-e ." in requirements:
            requirements.remove("-e .")
        
    return requirements
         
setup(
name='ML_ops',
version='0.0.1',
author='Mayank Ukani',
author_email='mayankukani12952@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt'),


)