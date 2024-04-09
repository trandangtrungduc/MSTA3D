import os
from setuptools import setup

def read(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    file = open(filepath, 'r')
    return file.read()

if __name__ == '__main__':
    setup(
        name='msta3d',
        version='1.0',
        description='MSTA3D: Multi-scale Twin-Attention for 3D Instance Segmentation',
        long_description=read('README.md'),
        author='Duc Tran',
        author_email='trandangtrungduc@seoultech.ac.kr/trandangtrungduc@gmail.com',
        packages=['msta3d'],
    )
