# Use an official Python runtime as a parent image
FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye


USER vscode

RUN sudo apt update
RUN sudo apt install python3-pip -y

# Install PyTorch and MLflow
RUN pip install --no-cache mlflow==2.12.1

# Install ODBC Drivers for SQL on Debian 11
# https://learn.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server?view=sql-server-ver16&tabs=debian18-install%2Calpine17-install%2Cdebian8-install%2Credhat7-13-install%2Crhel7-offline
RUN curl https://packages.microsoft.com/keys/microsoft.asc | sudo tee /etc/apt/trusted.gpg.d/microsoft.asc && \
    curl https://packages.microsoft.com/config/debian/11/prod.list | sudo tee /etc/apt/sources.list.d/mssql-release.list && \
    sudo apt-get update && \
    sudo ACCEPT_EULA=Y apt-get install -y msodbcsql18 && \
    sudo ACCEPT_EULA=Y apt-get install -y mssql-tools18 && \
    sudo apt-get  install -y unixodbc-dev && \
    sudo apt-get install -y libgssapi-krb5-2 && \
    sudo apt-get clean -y


RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.1.2-0-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm -rf ~/miniconda3/miniconda.sh && \
    ~/miniconda3/bin/conda init bash

# Fix issue #1 for non-login shells
RUN echo 'export PATH="/usr/local/python/current/bin:/usr/local/py-utils/bin:/usr/local/share/nvm/current/bin:/usr/local/bin:$PATH"' >> ~/.bashrc

# Add in ~/.local/bin - since we're installing some things from user context not root anymore
RUN echo 'export PATH="/home/vscode/.local/bin:$PATH"' >> ~/.bashrc

# Install Azure CLI the quick and easy way
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Install the Azure ML extension for Azure CLI
RUN az extension add -n ml
