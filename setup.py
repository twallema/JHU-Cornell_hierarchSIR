from setuptools import find_packages, setup, Extension
import pybind11

# boost installation paths
import platform
if platform.system() == 'Darwin':  # macOS
    boost_include_dir = '/opt/homebrew/Cellar/boost/1.87.0_1/include'
    boost_lib_dir = '/opt/homebrew/Cellar/boost/1.87.0_1/lib'
elif platform.system() == 'Linux':  # Ubuntu in GitHub Actions
    boost_include_dir = '/usr/include'  # Default path for Boost headers
    boost_lib_dir = '/usr/lib/x86_64-linux-gnu'  # Default path for Boost libraries
else:
    raise RuntimeError("Unsupported OS")

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
    description='A multi-strain SIR seasonal influenza model',
    author='Dr. Tijs W. Alleman, Johns Hopkins University, Cornell University',
    license='CC-BY-NC-SA',
    install_requires=['emcee','pySODM>=0.2.8'],
    ext_modules=ext_modules,
    python_requires='>3.12.0',
    extras_require={
        "develop":  ["pytest"]
    }
)





