// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package app

import (
	"context"
	"fmt"
	"log"

	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert/grpcapi"
	"github.com/nlpodyssey/spago/pkg/utils/grpcutils"
	"github.com/urfave/cli"
)

func newClientDiscriminateCommandFor(app *BertApp) cli.Command {
	return cli.Command{
		Name:        "discriminate",
		Usage:       "Perform linear discriminate analysis using BERT.",
		UsageText:   programName + " client discriminate --text=<value> [--address=<address>] [--tls-disable]",
		Description: "Run the " + programName + " client for linear discriminate analysis.",
		Flags:       newClientDiscriminateCommandFlagsFor(app),
		Action:      newClientDiscriminateCommandActionFor(app),
	}
}

func newClientDiscriminateCommandFlagsFor(app *BertApp) []cli.Flag {
	return []cli.Flag{
		cli.StringFlag{
			Name:        "address",
			Value:       "127.0.0.1:1976",
			Destination: &app.address,
		},
		cli.StringFlag{
			Name:        "text",
			Destination: &app.requestText,
			Required:    true,
		},
		cli.BoolFlag{
			Name:        "tls-disable ",
			Usage:       "Specifies that TLS is disabled.",
			Destination: &app.tlsDisable,
		},
	}
}

func newClientDiscriminateCommandActionFor(app *BertApp) func(c *cli.Context) {
	return func(c *cli.Context) {
		conn := grpcutils.OpenClientConnection(app.address, app.tlsDisable)
		cli := grpcapi.NewBERTClient(conn)

		resp, err := cli.Discriminate(context.Background(), &grpcapi.DiscriminateRequest{
			Text: app.requestText,
		})

		if err != nil {
			log.Fatalln(err)
		}

		fmt.Println(resp)
	}
}
