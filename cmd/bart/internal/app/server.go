// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package app

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/bpetokenizer"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/barthead"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartserver"
	"log"

	"github.com/urfave/cli"
)

func newServerCommandFor(app *BartApp) cli.Command {
	return cli.Command{
		Name:        "server",
		Usage:       "Run the " + programName + " as a server.",
		UsageText:   programName + " run --model=<path> [--grpc-address=<address>] [--tls-cert-file=<cert>] [--tls-key-file=<key>] [--tls-disable]",
		Description: "Run the " + programName + " indicating the model path (NOT the model file).",
		Flags:       newServerCommandFlagsFor(app),
		Action:      newServerCommandActionFor(app),
	}
}

func newServerCommandFlagsFor(app *BartApp) []cli.Flag {
	return []cli.Flag{
		cli.StringFlag{
			Name:        "grpc-address",
			Usage:       "Changes the bind address of the gRPC server.",
			Value:       "0.0.0.0:1976",
			Destination: &app.grpcAddress,
		},
		cli.StringFlag{
			Name:        "model, m",
			Required:    true,
			Usage:       "The path of the model to load.",
			Destination: &app.modelPath,
		},
		cli.StringFlag{
			Name:        "tls-cert-file",
			Usage:       "Specifies the path of the TLS certificate file.",
			Value:       "/etc/ssl/certs/spago/server.crt",
			Destination: &app.tlsCert,
		},
		cli.StringFlag{
			Name:        "tls-key-file",
			Usage:       "Specifies the path of the private key for the certificate.",
			Value:       "/etc/ssl/certs/spago/server.key",
			Destination: &app.tlsKey,
		},
		cli.BoolFlag{
			Name:        "tls-disable",
			Usage:       "Specifies that TLS is disabled.",
			Destination: &app.tlsDisable,
		},
	}
}

func newServerCommandActionFor(app *BartApp) func(c *cli.Context) {
	return func(c *cli.Context) {
		fmt.Printf("TLS Cert path is %s\n", app.tlsCert)
		fmt.Printf("TLS private key path is %s\n", app.tlsKey)

		tokenizer, err := bpetokenizer.NewFromModelFolder(app.modelPath)
		if err != nil {
			log.Fatal(err)
		}
		if tokenizer == nil {
			log.Fatal("expected BPETokenizer, actual nil")
		}

		model, err := barthead.LoadModelForSequenceClassification(app.modelPath)
		if err != nil {
			log.Fatal(err)
		}
		defer model.Close()
		fmt.Printf("Config: %+v\n", model.BART.Config)

		fmt.Printf("Start %s gRPC server listening on %s.\n", func() string {
			if app.tlsDisable {
				return "non-TLS"
			}
			return "TLS"
		}(), app.grpcAddress)

		server := bartserver.NewServer(model, tokenizer)
		server.StartDefaultServer(app.grpcAddress, app.tlsCert, app.tlsKey, app.tlsDisable)
	}
}
