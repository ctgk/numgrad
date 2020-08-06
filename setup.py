from setuptools import setup


__version__ = '0.0.1'


install_requires = []
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
    name='pyboilerplate',
    version=__version__,
    author='ctgk',
    author_email='r1135nj54w@gmail.com',
    description='pyboilerplate',

    python_requires='>=3',
    install_requires=install_requires,
    extras_require={
        'develop': develop_requires,
    },

    zip_safe=False,
)
