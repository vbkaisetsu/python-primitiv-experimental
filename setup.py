#!/usr/bin/env python3

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules=[
    Extension("primitiv.shape",
              sources=["primitiv/shape.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
    Extension("primitiv.tensor",
              sources=["primitiv/tensor.pyx", "primitiv/tensor_op.cpp"],
              language="c++",
              libraries=["primitiv"]
    ),
    Extension("primitiv.device",
              sources=["primitiv/device.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
    Extension("primitiv.cpu_device",
              sources=["primitiv/cpu_device.pyx"],
              language="c++",
              libraries=["primitiv"]
    )
]

setup(ext_modules = cythonize(ext_modules),
      packages = ["primitiv"])
