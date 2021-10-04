"""Setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject/blob/main/setup.py
"""


from setuptools import find_packages, setup


install_requires = [
    'numpy',
    'scipy',
]


setup(
    name='pygrad',
    version='0.1.0',
    author='ctgk',
    author_email='r1135nj54w@gmail.com',
    description='Simple gradient computation library in Python',

    packages=find_packages(
        exclude=('tests', 'tests.*'), include=('pygrad', 'pygrad.*')),
    python_requires='>=3.6',
    install_requires=install_requires,

    zip_safe=False,
)
