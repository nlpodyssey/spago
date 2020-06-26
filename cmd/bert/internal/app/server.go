// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package app

import (
	"fmt"
	"log"

	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert"
	"github.com/urfave/cli"
)

func newServerCommandFor(app *BertApp) cli.Command {
	return cli.Command{
		Name:        "server",
		Usage:       "Run the " + programName + " as a server.",
		UsageText:   programName + " run --model=<path> [--address=<address>] [--grpc-address=<address>] [--tls-cert-file=<cert>] [--tls-key-file=<key>] [--tls-disable]",
		Description: "Run the " + programName + " indicating the model path (NOT the model file).",
		Flags:       newServerCommandFlagsFor(app),
		Action:      newServerCommandActionFor(app),
	}
}

func newServerCommandFlagsFor(app *BertApp) []cli.Flag {
	return []cli.Flag{
		cli.StringFlag{
			Name:        "address",
			Usage:       "Changes the bind address of the server.",
			Value:       "0.0.0.0:1987",
			Destination: &app.address,
		},
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
			Name:        "tls-disable ",
			Usage:       "Specifies that TLS is disabled.",
			Destination: &app.tlsDisable,
		},
	}
}

func newServerCommandActionFor(app *BertApp) func(c *cli.Context) {
	return func(c *cli.Context) {
		fmt.Printf("TLS Cert path is %s\n", app.tlsCert)
		fmt.Printf("TLS private key path is %s\n", app.tlsKey)

		model, err := bert.LoadModel(app.modelPath)
		if err != nil {
			log.Fatalf("error during model loading (%v)\n", err)
		}
		fmt.Printf("Config: %+v\n", model.Config)

		fmt.Printf("Start %s HTTP server listening on %s.\n", func() string {
			if app.tlsDisable {
				return "non-TLS"
			}
			return "TLS"
		}(), app.address)
		fmt.Printf("Start %s gRPC server listening on %s.\n", func() string {
			if app.tlsDisable {
				return "non-TLS"
			}
			return "TLS"
		}(), app.grpcAddress)

		server := bert.NewServer(model)
		server.StartDefaultServer(app.address, app.grpcAddress, app.tlsCert, app.tlsKey, app.tlsDisable)
	}
}
