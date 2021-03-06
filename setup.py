# -*- coding: utf-8 -*-

import pathlib
from setuptools import setup, find_packages


ROOT_DIR = pathlib.Path(__file__).parent
ROOT_PKG = 'skcurve'


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
    python_requires='>=3.6, <4',
    install_requires=[
        'numpy',
        'scipy',
        'networkx',
        'csaps >=0.9.0, <1',
        'cached_property',
        'typing-extensions',
    ],
    extras_require={
        'plot': [
            'matplotlib',
        ],
        'docs': [
            'sphinx >=2.3',
            'numpydoc',
            'matplotlib',
        ],
        'examples': [
            'jupyter',
            'matplotlib',
        ],
        'tests': [
            'pytest',
            'coverage',
        ],
    },
    packages=find_packages(exclude=['tests', 'examples']),
    url='https://github.com/espdev/scikit-curve',
    project_urls={
        'Documentation': 'https://scikit-curve.readthedocs.io',
        'Code': 'https://github.com/espdev/scikit-curve',
        'Issue tracker': 'https://github.com/espdev/scikit-curve/issues',
    },
    license='BSD 3-Clause',
    author='Eugene Prilepin',
    author_email='esp.home@gmail.com',
    description='A toolkit to manipulate n-dimensional geometric curves in Python',
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
