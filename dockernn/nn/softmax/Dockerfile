FROM python:3.9-alpine AS base

RUN apk add --update gcc musl-dev linux-headers g++ gfortran

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt 

FROM python:3.9-alpine AS build-image
COPY --from=base /opt/venv /opt/venv
COPY ./softmax.py /softmax.py
RUN apk add util-linux
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 30003
CMD ["taskset","-c","0-3","python", "/softmax.py"]