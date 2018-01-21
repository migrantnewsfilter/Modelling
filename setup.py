import os
from setuptools import setup

setup(
    name="modelling",
    version='0.0.17',
    url="https://github.com/migrantnewsfilter/Modelling",

    py_modules=['modelling'],
    zip_safe=False,

    install_requires=[
        'pandas',
        'pymongo',
        'numpy',
        'sklearn',
        'scipy',
        'BeautifulSoup4',
        'html5lib'
    ]
)
