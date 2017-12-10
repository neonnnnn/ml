import numpy as np
from Cython.Distutils import build_ext
from setuptools import setup, Extension

ext_modules = [
    Extension('canova', sources=['canova.pyx'],
              include_dirs=[np.get_include()])
]
setup(
    name='ANOVA Kernel',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext}
)