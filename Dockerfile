FROM tensorflow/tensorflow:1.12.0-py3

RUN apt-get update \
    && apt-get install -y libsm6 libxrender-dev libxext6

ADD $PWD/requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

CMD ["/bin/bash"]