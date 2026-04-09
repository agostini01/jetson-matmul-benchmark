FROM dustynv/pytorch:2.1-r36.2.0 AS base

ARG DEBIAN_FRONTEND=noninteractive

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt update &&  \
#     apt -y install --no-install-recommends <your-package-list-here>

# Do NOT modify UID from 1000
# vscode updates this UID to the user building the container
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt update \
    && apt install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && apt autoremove -y \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Install additional OS packages and clean apt metadata in the same layer.
RUN apt update \
    && apt install -y --no-install-recommends clang-format \
    && apt autoremove -y \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/clang-format-14 /usr/bin/clang-format

USER $USERNAME
