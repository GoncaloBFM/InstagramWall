FROM python:3.7-slim-stretch

RUN pip install \
	pandas \
	tqdm \

RUN apt-get update && apt-get install -y vim

RUN mkdir -p /InstagramWall/{data, seek_result, thumbs}
WORKDIR /InstagramWall/

CMD ["/bin/bash"]
#CMD [ "train.py" ]