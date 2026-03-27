"""Build the Cython PUCT extension.

Usage: python learned_eval/setup_puct.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "learned_eval._puct_cy",
        ["learned_eval/_puct_cy.pyx"],
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
    }),
)
