from setuptools import setup, find_packages


install_requires = [
    'numpy',
    'scipy',
]
develop_requires = [
    'autopep8',
    'flake8',
    'pep8-naming',
    'pre-commit',
    'pytest',
    'sphinx',
    'sphinx_rtd_theme',
    'livereload',

    'matplotlib',
    'scikit-learn',
    'tqdm',
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
    extras_require={
        'develop': develop_requires,
    },

    zip_safe=False,
)
