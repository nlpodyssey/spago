# Copyright 2020 spaGO Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


# Builder container builds the demo programs for named entities
# recognition (ner-server), model importing (hugging_face_importer),
# and question answering (bert_server). The binaries are copied into
# the same runtime container below. The version of Go given in the
# image tag must match the version of Go in go.mod.
FROM golang:1.14-alpine as Builder

# Some of the Go packages used by spaGo require gcc.
RUN set -eux; \
	apk add --no-cache --virtual .build-deps \
		gcc \
		musl-dev \
        ;

RUN mkdir /build
ADD . /build/
WORKDIR /build
RUN go mod download
RUN GOOS=linux GOARCH=amd64 go build -o ner-server cmd/ner/main.go
RUN GOOS=linux GOARCH=amd64 go build -o hugging_face_importer cmd/huggingfaceimporter/main.go
RUN GOOS=linux GOARCH=amd64 go build -o bert_server cmd/bert/main.go


# The definition of the runtime container now follows, and it contains
# demo programs for named entities recognition (ner-server), model
# importing (hugging_face_importer), and question answering
# (bert_server). The version of Alpine given in the image tag should
# match the version of Alpine used in the Builder image.
FROM alpine:3.12

RUN apk add --no-cache \
		ca-certificates \
		;

ENV GOOS linux
ENV GOARCH amd64

RUN addgroup -S spago && adduser -S spago -G spago

COPY --chown=spago:spago docker-entrypoint.sh /home/spago/
RUN chmod 0755 /home/spago/docker-entrypoint.sh

COPY --chown=spago:spago --from=Builder /build/* /home/spago/

USER spago
WORKDIR /home/spago
ENTRYPOINT ["/home/spago/docker-entrypoint.sh"]
CMD ["help"]
