#!/usr/bin/env/python
"""Installation script
"""

import setuptools
import versioneer


version = versioneer.get_version()
cmdclass = versioneer.get_cmdclass()


setuptools.setup(
    version=version,
    cmdclass=cmdclass
)


## import os
## 
## try:
##     from setuptools import setup
## except ImportError:
##     from distutils.core import setup
## 
## with open('requirements.txt') as f:
##     install_requires = f.read().splitlines()
## 
## LONG_DESCRIPTION = """
## pystareplotlib is a simple plotting mechanism arising from the need to visualize and explore STARE-based methods of Earth Science data analysis."""
## 
## # get all data dirs in the datasets module
## data_files = []
## 
## setup(
##     name="pystareplotlib",
##     version='0.1.0',
##     description="STARE Hello World Plotting",
##     license="MIT",
##     author="Michael Rilee",
##     author_email="mike@rilee.net",
##     url="https://github.com/SpatioTemporal/pystareplotlib",
##     long_description=LONG_DESCRIPTION,
##     packages=[
##         "pystareplotlib"
##     ],
##     python_requires=">=3.5",
##     install_requires=install_requires,
## ) 






