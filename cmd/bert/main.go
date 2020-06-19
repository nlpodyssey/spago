// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"log"
	"os"

	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert"
	"github.com/urfave/cli"
)

const (
	programName = "bert_server"
)

func main() {
	app := newBertServerApp()
	app.Run(os.Args)
}

type bertServerApp struct {
	*cli.App
	address    string
	modelPath  string
	tlsCert    string
	tlsKey     string
	tlsDisable bool
}

func newBertServerApp() *bertServerApp {
	app := &bertServerApp{
		App: cli.NewApp(),
	}
	app.Name = programName
	app.Usage = "A demo for question-answering based on BERT."
	app.Commands = []cli.Command{
		newRunCommandFor(app),
	}
	return app
}

func newRunCommandFor(app *bertServerApp) cli.Command {
	return cli.Command{
		Name:        "run",
		Usage:       "Run the " + programName + ".",
		UsageText:   programName + " run --model=<path> [--address=<address>] [--tls-cert-file=<cert>] [--tls-key-file=<key>] [--tls-disable]",
		Description: "Run the " + programName + " indicating the model path (NOT the model file).",
		Flags:       newRunCommandFlagsFor(app),
		Action:      newRunCommandActionFor(app),
	}
}

func newRunCommandFlagsFor(app *bertServerApp) []cli.Flag {
	return []cli.Flag{
		cli.StringFlag{
			Name:        "address",
			Usage:       "Changes the bind address of the server.",
			Value:       "0.0.0.0:1987",
			Destination: &app.address,
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

func newRunCommandActionFor(app *bertServerApp) func(c *cli.Context) {
	return func(c *cli.Context) {
		fmt.Printf("TLS Cert path is %s\n", app.tlsCert)
		fmt.Printf("TLS private key path is %s\n", app.tlsKey)

		model, err := bert.LoadModel(app.modelPath)
		if err != nil {
			log.Fatalf("error during model loading (%v)\n", err)
		}
		fmt.Printf("Config: %+v\n", model.Config)

		fmt.Printf("Start %s server listening on %s.\n", func() string {
			if app.tlsDisable {
				return "non-TLS"
			}
			return "TLS"
		}(), app.address)
		server := bert.NewServer(model)
		server.StartDefaultServer(app.address, app.tlsCert, app.tlsKey, app.tlsDisable)
	}
}
