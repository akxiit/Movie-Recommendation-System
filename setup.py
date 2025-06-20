from setuptools import setup, find_packages
from typing import List


HYPEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:

    with open(file_path) as file_obj:

        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='Movie-Recommendation-System',
    version='0.0.1',
    author='Akshit Vaghasiya',
    author_email='akshitvaghasiya504@gmail.com',
    find_packages=find_packages(),
    install_dependencies = get_requirements('requirements.txt')

)