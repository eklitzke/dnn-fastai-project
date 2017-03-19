from setuptools import setup, find_packages


setup(name='dnn-fastai-project',
      version='1.0',
      packages=find_packages(),
      install_requires=[
          'keras',
          'numpy',
      ],
      entry_points={
          'console_scripts': [
              'vidextend = vidextend.main:main',
          ]})
