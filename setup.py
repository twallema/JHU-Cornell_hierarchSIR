from setuptools import setup, Extension
import platform
import pybind11

# This setup file contains the dynamic parts of `hierarchSIR` installer, while `pyproject.toml` contains the static parts

# find the C++ Boost integration module
def find_boost_include():
    if platform.system() == 'Darwin':
        return '/opt/homebrew/Cellar/boost/1.87.0_1/include'
    elif platform.system() == 'Linux':
        return '/usr/include'
    else:
        raise RuntimeError("Unsupported OS")

def find_boost_lib():
    if platform.system() == 'Darwin':
        return '/opt/homebrew/Cellar/boost/1.87.0_1/lib'
    elif platform.system() == 'Linux':
        return '/usr/lib/x86_64-linux-gnu'
    else:
        raise RuntimeError("Unsupported OS")

# define the C++ module we want to bind to Python
ext_modules = [
    Extension(
        "hierarchSIR.sir_model",
        sources=["src/hierarchSIR/sir_model.cpp"],
        include_dirs=[pybind11.get_include(), find_boost_include()],
        library_dirs=[find_boost_lib()],
        libraries=["boost_system"],
        language="c++",
        extra_compile_args=["-std=c++14"]
    )
]

setup(
    ext_modules=ext_modules
)
