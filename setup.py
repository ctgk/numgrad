"""Setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject/blob/main/setup.py
"""

import codecs
import os

from setuptools import find_packages, setup


install_requires = [
    'numpy',
    'scipy',
]


def _read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def _get_version(rel_path):
    for line in _read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name='numflow',
    version=_get_version('numflow/_version.py'),
    author='ctgk',
    author_email='r1135nj54w@gmail.com',
    description='Simple gradient computation library in Python',

    packages=find_packages(
        exclude=('tests', 'tests.*'),
        include=('numflow', 'numflow.*'),
    ),
    python_requires='>=3.6',
    install_requires=install_requires,

    zip_safe=False,
)
