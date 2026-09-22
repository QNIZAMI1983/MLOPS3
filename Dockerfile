FROM ubuntu:latest
LABEL authors="qniza"

ENTRYPOINT ["top", "-b"]