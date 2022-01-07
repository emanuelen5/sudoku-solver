from distutils.core import setup, Extension
import numpy

module = Extension('hough', sources = ['hough.c'], include_dirs=[numpy.get_include()])

setup(
    name = 'sudoku',
    version = '0.1.0',
    description = 'Package for solving sudoku:s',
    ext_modules = [module],
)