FROM andrewosh/binder-base

MAINTAINER Ákos Laczkó <akos.laczko88@gmail.com>

USER root

RUN apt-get update
RUN apt-get -y install ttf-freefont

RUN pip install numpy scipy pandas matplotlib seaborn
