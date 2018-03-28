FROM debian:latest

RUN apt-get -y update && apt-get install -y git python3-pip python3-dev python3-tk vim procps curl

WORKDIR ekholabs/face-classifier
ADD REQUIREMENTS.txt /ekholabs/face-classifier/REQUIREMENTS.txt
RUN pip3 install -r REQUIREMENTS.txt

ADD . /ekholabs/face-classifier

ENV PYTHONPATH=$PYTHONPATH:src
ENV FACE_CLASSIFIER_PORT=8084
EXPOSE $FACE_CLASSIFIER_PORT

ENTRYPOINT ["python3"]
CMD ["src/web/faces.py"]
