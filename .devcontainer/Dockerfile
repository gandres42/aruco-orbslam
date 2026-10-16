FROM osrf/ros:lyrical-desktop-full
ARG USERNAME=USERNAME
ARG CONTAINER_ID=CONTAINER_ID
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ENV DEBIAN_FRONTEND=noninteractive

# create non-root user and add to sudo
RUN if id -u $USER_UID ; then userdel `id -un $USER_UID` ; fi
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# install packages
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y \
    git \
    wget \
    python3-pip \
    python-is-python3 \
    software-properties-common \
    libgl1 \
    ros-lyrical-cv-bridge \
    libboost-all-dev \
    libcppunit-dev \
    ros-lyrical-foxglove-bridge \
    ros-lyrical-rmw-zenoh-cpp \
    python3-colcon-common-extensions \
    clangd \
    jq \
    ssh \
    libepoxy-dev

# install pangolin
RUN cd /tmp && \
    git clone https://github.com/stevenlovegrove/Pangolin && \
    cd Pangolin && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j8 && \
    make install

# install orb-slam3
RUN cd /tmp && \
    git clone https://github.com/gandres42/ORB_SLAM3.git && \
    cd ORB_SLAM3 && \
    ./build.sh && \
    cd build && \
    make install && \
    ldconfig

# dev environment setup
RUN chsh -s /bin/bash $USERNAME
USER $USERNAME
RUN echo "source /home/$USERNAME/.dotfiles/setup.bash" >> ~/.bashrc
ENV CONTAINER_ID $CONTAINER_ID