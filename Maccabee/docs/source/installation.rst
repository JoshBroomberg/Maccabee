Installing Maccabee
===================

Standard Python Installation
----------------------------

Maccabee is available as a python package. To install it, run the following command at the command line:

.. code-block:: bash

    pip install maccabee

Installing optional R dependencies
----------------------------------

If you plan to work with R models, install the Maccabee package along with optional R dependencies by running:

.. code-block:: bash

    pip install maccabee[r]

Docker Image
------------

There is a Docker image available with Maccabee and all dependencies pre-installed (including all R dependencies). This docker image is primarily designed to run a Jupyter notebook server.

The image is hosted on Dockerhub under the name *maccabeebench/jupyter_maccabee:latest* (for the latest version). To run a notebook server using this image, simply the command below. It will mount your current working directory at `~/work` and make the server accessible (without an authentication requirement) at `localhost:8888`.

.. code-block:: bash

    docker run --rm -p 8888:8888 \
      -v $(PWD):/home/jovyan/work \
      maccabeebench/jupyter_maccabee:latest \
      start.sh jupyter notebook \
      --NotebookApp.token='' \
      --NotebookApp.notebook_dir='~/work' \
      --NotebookApp.ip='0.0.0.0'
