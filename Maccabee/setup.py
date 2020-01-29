import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='maccabee',
     version='0.0.12',
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
          'POT',
          'pyyaml',
      ],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     python_requires='>=3.6',
     include_package_data=True
 )
