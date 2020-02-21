## Start from a core stack version
FROM jupyter/datascience-notebook

## Root commands
USER root

RUN sudo apt-get -y update && \
  sudo apt-get install texlive-generic-extra && \
  sudo apt-get install -y rsync && \
  sudo apt-get install -y software-properties-common && \
  sudo apt-get install -y libcurl4-openssl-dev libssl-dev && \
  sudo apt-get install -y libxml2 libxml2-dev && \
  sudo apt-get clean

# Configure R 3.6
# https://cran.r-project.org/bin/linux/ubuntu/
RUN echo "deb https://cloud.r-project.org/bin/linux/ubuntu xenial-cran35/" >> /etc/apt/sources.list
RUN echo "deb http://mirror.enzu.com/ubuntu/ bionic-backports main restricted universe" >> /etc/apt/sources.list
RUN sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN sudo add-apt-repository ppa:marutter/c2d4u3.5 && \
  sudo add-apt-repository "deb http://ftp.ubuntu-tw.net/ubuntu/ bionic main" && \
  sudo add-apt-repository "deb http://ftp.ubuntu-tw.net/ubuntu/ xenial main" && \
  sudo apt update

RUN sudo apt-get install -y libcurl3 libpng12-0 libreadline6 && \
  sudo apt-get install -y r-base r-base-core r-recommended && \
#  sudo apt-get install -y r-base-dev && \
#  sudo apt install -y r-cran-devtools r-cran-roxygen2 && \
  sudo apt-get clean

## User Commands: set Jovyan as user for rest of script
USER $NB_USER

## Upgrade Jupyter Notebook/Lab

RUN pip install --upgrade notebook && \
  ## Download and enable Notebook extensions
  pip install --upgrade jupyter_contrib_nbextensions && \
  jupyter contrib nbextension install --user && \
  ## Install Google Colab Server Extension
  pip install jupyter_http_over_ws && \
  jupyter serverextension enable --py jupyter_http_over_ws && \
  ## Install NBConvert Export Formatting
  pip install nb_pdf_template && \
  python -m nb_pdf_template.install && \
  ## Install themes
  pip install jupyterthemes && \
  jt -T -t grade3

# Initialize plotting config at start up
# TODO: this doesn't work right now.
# COPY startup.ipy .ipython/profile_default/startup/

## Install python packages from requirements.txt file
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt

## Set destination for tmp R library installation.
RUN mkdir -p /home/jovyan/lib/R/packages
ENV R_LIBS_USER=/home/jovyan/lib/R/packages

## Install R packages.
# USER root
COPY r_packages.R /tmp
RUN Rscript /tmp/r_packages.R
