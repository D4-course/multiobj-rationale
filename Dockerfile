# syntax=docker/dockerfile:1s
# Define base image
FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml
# RUN conda init bash
# Override default shell and use bash
SHELL ["conda", "run", "-n", "env", "/bin/bash", "-c"]

# ENTRYPOINT ["./entrypoint.sh"]

# SHELL ["conda", "activate", "env"]
# RUN conda activate env

SHELL ["/bin/bash","-c"]
RUN conda init
RUN echo 'conda activate env' >> ~/.bashrc
RUN bash ~/.bashrc
RUN conda install -c conda-forge fastapi
RUN conda install uvicorn
RUN conda install -c conda-forge python-multipart
EXPOSE 8000
