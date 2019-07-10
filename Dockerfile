FROM python:3.7-slim-stretch

RUN apt-get update && apt-get install -y vim g++

RUN pip install \
	pandas \
	tqdm \
	selenium \
	tables \
	tensorflow \
	requests \
	scikit-image \
	nmslib \
	sklearn

WORKDIR /InstagramWall/

CMD ["/bin/bash"]
#CMD [ "train.py" ]