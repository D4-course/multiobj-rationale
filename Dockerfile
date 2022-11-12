# syntax=docker/dockerfile:1s
# Define base image
FROM continuumio/miniconda3

WORKDIR /APP

COPY environment.yml .
RUN conda env create -f environment.yml
 
# Override default shell and use bash
SHELL ["conda", "run", "-n", "env", "/bin/bash", "-c"]

# SHELL ["conda", "activate", "rationale"]



