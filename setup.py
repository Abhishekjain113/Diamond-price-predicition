from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirments(file_path:str)->List[str]:
    requirment=[]
    with open(file_path) as file_obj:
        requirment=file_obj.readline()
        requirment=[req.replace('\n','') for req in requirment]
        if HYPEN_E_DOT in requirment:
            requirment.remove(HYPEN_E_DOT)
        return requirment




setup(
    name='DiamondPricePrediction',
    version='0.0.1',
    author='jainabhishek',
    author_email='jainabhishek1117@gmai.com',
    install_require=get_requirments('requirment.txt'),
    packages=find_packages()

)