"""Setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject/blob/main/setup.py
"""

import pathlib

from setuptools import find_packages, setup


install_requires = [
    'numpy',
    'scipy',
]


def _read(rel_path):
    here = pathlib.Path(__file__).parent
    return (here / rel_path).read_text()


def _get_version(rel_path):
    for line in _read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name='numgrad',
    version=_get_version('numgrad/_version.py'),
    author='ctgk',
    author_email='r1135nj54w@gmail.com',
    url='https://github.com/ctgk/numgrad',
    description='Simple gradient computation library in Python',
    long_description=_read('README.md'),
    long_description_content_type='text/markdown',
    license='MIT',

    packages=find_packages(
        exclude=('tests', 'tests.*'),
        include=('numgrad', 'numgrad.*'),
    ),
    python_requires='>=3.8',
    install_requires=install_requires,

    zip_safe=False,
)
