"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="ml-impute",  # Required
    version="0.0.4",  # Required
    description="A package for synthetic data generation for imputation using single and multiple imputation methods.",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="https://github.com/JoshWeiner/ml-impute",  # Optional
    author="Joshua Weiner",  # Optional
    author_email="joshuaweinernyc@gmail.com",  # Optional
    license='MIT',
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords=[ "machine learning", "synthetic data", "imputation", "singular imputation", "multiple imputation"],
    # package_dir={"": "mpute"},  # Optional
    # packages=find_packages(where="mpute"),  # Required
    python_requires=">=3.6, <4",
    install_requires=["pandas", "numpy", "jax", "jaxlib"],  # Optional
    packages=["mpute"],
    # extras_require={  # Optional
    #     "dev": ["check-manifest"]
    #     # "test": ["coverage"],
    # },
    package_data={  # Optional
        # "sample": ["package_data.dat"],
    },
    # data_files=[("my_data", ["data/data_file"])],  # Optional
    # entry_points={  # Optional
    #     "console_scripts": [
    #         "sample=sample:main",
    #     ],
    # },
    project_urls={  # Optional
        "Bug Reports": "https://github.com/JoshWeiner/ml-impute/issues",
        "Funding": "https://donate.pypi.org",
        "Say Thanks!": "http://saythanks.io/to/example",
        "Source": "https://github.com/JoshWeiner/ml-impute/",
    },
)