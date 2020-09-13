from setuptools import setup
import setuptools

REQUIRES = ["numpy", "matplotlib", "gym"]
setup(name='learn_seq',
      version='0.0.1',
      packages=setuptools.find_packages(),
      install_requires=REQUIRES
)
