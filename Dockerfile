# Copyright 2020 spaGO Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


# Builder container builds the demo programs for named entities
# recognition (ner-server), Hugging Face model importing
# (hugging-face-importer), BERT (bert-server), BART (bert-server).
# The binaries are then copied into the same runtime container below.
# The version of Go given in the image tag must match the version of Go in go.mod.
# The binaries have all been statically linked, and they were built without cgo.
FROM golang:1.15.6-alpine3.12 as Builder

# Some of the Go packages used by spaGo require gcc. OpenSSL is used
# to generate a self-signed cert in order to test the Docker image.
# The packages ca-certificates is needed in order to run the servers
# using TLS.
RUN set -eux; \
	apk add --no-cache --virtual .build-deps \
		ca-certificates \
		gcc \
		musl-dev \
		openssl \
        ;

# The spago user is created so that the servers can be run
# with limited privileges.
RUN adduser -S spago

# Build statically linked Go binaries without CGO.
RUN mkdir /build
ADD . /build/
WORKDIR /build
RUN go mod download
RUN GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -ldflags="-extldflags=-static" -o docker-entrypoint docker-entrypoint.go

# A self-signed certificate and private key is generated so that the
# servers can easily support TLS without requiring the user to
# generate their own certificates.
RUN mkdir /etc/ssl/certs/spago \
	&& openssl req \
		-x509 \
		-nodes \
		-newkey rsa:2048 \
		-keyout /etc/ssl/certs/spago/server.key \
		-out /etc/ssl/certs/spago/server.crt \
		-days 3650 \
		-subj "/C=IT/ST=Piedmont/L=Torino/O=NLP Odyssey/OU=spaGo/emailAddress=matteogrella@gmail.com/CN=*" \
	&& chmod +r /etc/ssl/certs/spago/server.key \
	;


# The definition of the runtime container now follows.
FROM scratch

# Copy the user info from the Builder container.
COPY --from=Builder /etc/passwd /etc/passwd
USER spago

# Copy the CA certs and the self-signed cert.
COPY --from=Builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=Builder /etc/ssl/certs/spago/server.crt /etc/ssl/certs/spago/server.crt
COPY --from=Builder /etc/ssl/certs/spago/server.key /etc/ssl/certs/spago/server.key

# Copy the docker entrypoint from the Builder container.
COPY --from=Builder /build/docker-entrypoint /docker-entrypoint

# Setup the environment and run the script docker-entrypoint.sh so
# that a help screen is printed to the user when no commands are given.
ENV GOOS linux
ENV GOARCH amd64
ENTRYPOINT ["/docker-entrypoint"]
CMD ["help"]
