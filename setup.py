import pybind11
from setuptools import find_packages, setup, Extension

# boost installation paths
boost_include_dir = '/opt/homebrew/Cellar/boost/1.87.0_1/include' # brew info boost
boost_lib_dir = '/opt/homebrew/Cellar/boost/1.87.0_1/lib'

# point toward C++ model
ext_modules = [
    Extension(
        "hierarchSIR.sir_model",
        ["src/hierarchSIR/sir_model.cpp"],
        include_dirs=[
            pybind11.get_include(),
            boost_include_dir,  # Add Boost include path
        ],
        libraries=["boost_system"],  # Link Boost
        library_dirs=[boost_lib_dir],  # Add Boost library path
        language="c++",
        extra_compile_args=["-std=c++14"],  # Ensure C++14 support
    ),
]

# installs the Python package
setup(
    name='hierarchSIR',
    packages=find_packages("src", exclude=["*.tests"]),
    package_dir={'': 'src'},
    version='0.0',
    description='An SIR influenza model for the USA',
    author='Dr. Tijs W. Alleman, Johns Hopkins University, Cornell University',
    license='CC-BY-NC-SA',
    install_requires=['emcee','pySODM==0.2.6'],
    ext_modules=ext_modules
)