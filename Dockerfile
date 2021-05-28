# --------- How to build and run --------- #
#  docker build -t flask_server:0.1 .      #
#  docker run -p 5000:5000 flask_server:0.1    #
# ---------------------------------------- #

FROM ubuntu:18.04

RUN apt-get update

# Installing python and git
RUN apt-get install -y python3.8 python3-pip python3.8-dev
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install -y git
RUN python3.8 -m pip install --upgrade pip

# Installing python modules
RUN python3.8 -m pip install flask
RUN python3.8 -m pip install numpy
RUN python3.8 -m pip install tensorflow
RUN python3.8 -m pip install Pillow


WORKDIR /home
RUN git clone https://github.com/raresgithub/MDS_PROJECT.git
WORKDIR MDS_PROJECT
	
# Run the script
CMD python3.8 main.py
