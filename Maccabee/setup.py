import setuptools
import os

setup_path = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(setup_path, "./README.md")

with open(readme_path, "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='maccabee',
     version='1.0.2',
     scripts=[],
     author="Josh Broomberg",
     author_email="joshbroomberg@gmail.com",
     description="Causal ML benchmarking and development tools",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/JoshBroomberg/capstone/tree/master/DataGeneration/CauseML",
     packages=setuptools.find_packages(),
     install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'sympy',
          'scikit-learn',
          'POT',
          'pyyaml',
          'threadpoolctl'
      ],
      extras_require = {
        "docs": [
          "sphinx",
          "sphinx-autobuild",
          "sphinx_rtd_theme",
          "nbsphinx"
        ],
        "r": [
            "rpy2"
        ]
     },
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     python_requires='>=3.6',
     include_package_data=True
 )
