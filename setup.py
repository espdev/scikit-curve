# -*- coding: utf-8 -*-

import pathlib
from setuptools import setup, find_packages


ROOT_DIR = pathlib.Path(__file__).parent
ROOT_PKG = 'curve'


def get_version():
    version_info = {}
    version_file = ROOT_DIR / ROOT_PKG / '_version.py'
    with version_file.open() as f:
        exec(f.read(), version_info)
    return version_info['__version__']


def get_long_description():
    readme_file = ROOT_DIR / 'README.md'
    return readme_file.read_text(encoding='utf-8')


setup(
    name='scikit-curve',
    version=get_version(),
    python_requires='>=3.5,<4',
    install_requires=[
        'numpy',
        'scipy',
    ],
    packages=find_packages(exclude=['tests', '*.tests', '*.tests.*', 'tests.*']),
    url='https://github.com/espdev/scikit-curve',
    license='BSD 3-Clause',
    author='Eugene Prilepin',
    author_email='esp.home@gmail.com',
    description='A set of tools to manipule curves in Python',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
)
