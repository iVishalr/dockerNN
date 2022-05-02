FROM python:3.9-alpine AS base

RUN apk add --update gcc musl-dev linux-headers g++ gfortran

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt 

FROM python:3.9-alpine AS build-image
COPY --from=base /opt/venv /opt/venv
COPY ./cross_entropy_loss.py /cross_entropy_loss.py
RUN apk add util-linux
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 30006
CMD ["taskset","-c","0-3","python", "-u", "/cross_entropy_loss.py"]