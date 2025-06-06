from setuptools import setup, Extension
import platform
import pybind11

# This setup file contains the dynamic parts of `hierarchSIR` installer, while `pyproject.toml` contains the static parts

# find the C++ Boost integration module
def find_boost_include():
    # If BOOST_ROOT is set in the environment (likely cluster with module loaded)
    boost_root = os.environ.get("BOOST_ROOT")
    if boost_root:
        return os.path.join(boost_root, "include")
    # Else local Linux path
    if platform.system() == 'Linux':
        return '/usr/include'
    # Or macOS
    elif platform.system() == 'Darwin':
        return '/opt/homebrew/Cellar/boost/1.87.0_1/include'
    else:
        raise RuntimeError("Unsupported OS")

def find_boost_lib():
    # If BOOST_ROOT is set in the environment (likely cluster with module loaded)
    boost_root = os.environ.get("BOOST_ROOT")
    if boost_root:
        return os.path.join(boost_root, "lib")
    # Else local Linux path
    if platform.system() == 'Linux':
        return '/usr/lib/x86_64-linux-gnu'
    # Or macOS
    elif platform.system() == 'Darwin':
        return '/opt/homebrew/Cellar/boost/1.87.0_1/lib'
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




import os
import platform
from setuptools import setup, Extension
import pybind11

def find_boost_include():
    # If BOOST_ROOT is set in the environment (likely cluster with module loaded)
    boost_root = os.environ.get("BOOST_ROOT")
    if boost_root:
        return os.path.join(boost_root, "include")

    # Else local Linux path
    if platform.system() == 'Linux':
        return '/usr/include'

    # macOS paths unchanged
    elif platform.system() == 'Darwin':
        return '/opt/homebrew/Cellar/boost/1.87.0_1/include'

    else:
        raise RuntimeError("Unsupported OS")

def find_boost_lib():
    boost_root = os.environ.get("BOOST_ROOT")
    if boost_root:
        return os.path.join(boost_root, "lib")

    if platform.system() == 'Linux':
        return '/usr/lib/x86_64-linux-gnu'

    elif platform.system() == 'Darwin':
        return '/opt/homebrew/Cellar/boost/1.87.0_1/lib'

    else:
        raise RuntimeError("Unsupported OS")

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
