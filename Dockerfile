FROM python:3.7-slim-stretch

RUN pip install \
	pandas \
	tqdm \
	selenium \
	tables \
	tensorflow \
	requests \
	scikit-image

RUN apt-get update && apt-get install -y vim

WORKDIR /InstagramWall/

CMD ["/bin/bash"]
#CMD [ "train.py" ]