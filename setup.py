from setuptools import setup


__version__ = '0.0.1'


install_requires = [
    'numpy',
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
]


setup(
    name='pygrad',
    version=__version__,
    author='ctgk',
    author_email='r1135nj54w@gmail.com',
    description='pygrad',

    python_requires='>=3.6',
    install_requires=install_requires,
    extras_require={
        'develop': develop_requires,
    },

    zip_safe=False,
)
