from setuptools import setup, find_packages

setup(
    name='tracker',
    version='0.1dev1',
    description='Experimental code for CSRF road tracking',
    packages=find_packages(exclude=['tests.*', 'tests']),
)
