FROM andrewosh/binder-base

MAINTAINER Andrew Osheroff <andrewosh@gmail.com>

USER root

RUN apt-get update
RUN pip install -y numpy scipy pandas matplotlib
