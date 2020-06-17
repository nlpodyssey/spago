// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"log"
	"os"

	"github.com/docopt/docopt-go"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert"
)

func main() {
	usage := `Usage: bert_server --model=<path> [--address=<address>] [--tls-cert-file=<cert>] [--tls-key-file=<key>] [--tls-disable]

Options:
     --address=<bind address>    Changes the bind address of the server. [default: 0.0.0.0:1987]
  -m --model=<path>              The path of the model to load.
	 --tls-cert-file=<cert>      Specifies the path of the TLS certificate file. [default: /etc/ssl/certs/spago/server.crt]
	 --tls-key-file=<key>        Specifies the path of the private key for the certificate. [default: /etc/ssl/certs/spago/server.key]
	 --tls-disable               Specifies that TLS is disabled.
  -h --help                      Show this screen.`

	arguments, err := docopt.ParseDoc(usage)
	if err != nil {
		log.Fatal(err)
	}

	address, err := arguments.String("--address")
	if err != nil {
		log.Fatal(err)
	}

	tlsCert, err := arguments.String("--tls-cert-file")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("TLS Cert path is %s\n", tlsCert)

	tlsKey, err := arguments.String("--tls-key-file")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("TLS private key path is %s\n", tlsKey)

	tlsDisable, err := arguments.Bool("--tls-disable")
	if err != nil {
		log.Fatal(err)
	}

	modelPath := mustStr(arguments.String("--model"))
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		log.Fatal(err)
	}

	model, err := bert.LoadModel(modelPath)
	if err != nil {
		log.Fatal(fmt.Sprintf("error during model loading (%s)", err.Error()))
	}
	fmt.Printf("Config: %+v\n", model.Config)

	fmt.Printf("Start %s server listening on %s.\n", func() string {
		if tlsDisable {
			return "non-TLS"
		}
		return "TLS"
	}(), address)
	server := bert.NewServer(model)
	server.StartDefaultServer(address, tlsCert, tlsKey, tlsDisable)
}

func mustStr(value string, _ error) string {
	return value
}
