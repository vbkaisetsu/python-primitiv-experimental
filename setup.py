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
    ),
    #Extension("primitiv.cuda_device",
              #sources=["primitiv/cuda_device.pyx"],
              #language="c++",
              #libraries=["primitiv"]
    #),
    Extension("primitiv.function",
              sources=["primitiv/function.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
    Extension("primitiv.functions.function_impl",
              sources=["primitiv/functions/function_impl.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
    Extension("primitiv.parameter",
              sources=["primitiv/parameter.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
    Extension("primitiv.initializer",
              sources=["primitiv/initializer.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
    Extension("primitiv.initializers.initializer_impl",
              sources=["primitiv/initializers/initializer_impl.pyx"],
              language="c++",
              libraries=["primitiv"]
    ),
]

setup(
    ext_modules = cythonize(ext_modules),
    packages = ["primitiv",
                "primitiv.functions",
                "primitiv.initializers"]
)
