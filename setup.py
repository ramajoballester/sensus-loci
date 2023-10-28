from setuptools import setup, find_packages

name = 'sensus'

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name=name,
    version='0.1',
    author='Álvaro Ramajo Ballester',
    author_email='aramajo@pa.uc3m.es',
    description='Package description',
    # packages=find_packages(where='./{}'.format(name), exclude=['docs', 'data']),
    packages=find_packages(),
    
    long_description=open('README.md').read(),
    install_requires=install_requires
)