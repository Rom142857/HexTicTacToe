"""Build the C++ extension modules.

Usage:
    python setup.py build_ext --inplace
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

common_args = ["-O3", "-march=native", "-DNDEBUG"]

setup(
    name="hex_cpp_bots",
    ext_modules=[
        Pybind11Extension("ai_cpp", ["ai_cpp.cpp"],
                          cxx_std=17, extra_compile_args=common_args),
        Pybind11Extension("ai_cpp_og", ["ai_cpp_og.cpp"],
                          cxx_std=17, extra_compile_args=common_args),
    ],
    cmdclass={"build_ext": build_ext},
)
