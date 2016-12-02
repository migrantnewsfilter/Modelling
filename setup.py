import os
from setuptools import setup

setup(
    name="modelling",
    version='0.0.3',
    url="https://github.com/migrantnewsfilter/Modelling",

    py_modules=['modelling'],
    zip_safe=False,

    install_requires=[
        'pandas',
        'MongoClient',
        'numpy',
        'sklearn',
        'scipy'
    ]
)
