// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package app

import (
	"context"
	"log"

	"github.com/nlpodyssey/spago/cmd/clientutils"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartserver/grpcapi"
	"github.com/urfave/cli"
)

func newClientClassifyCommandFor(app *BartApp) cli.Command {
	return cli.Command{
		Name:        "classify",
		Usage:       "Perform text classification using BART.",
		Description: "Run the " + programName + " client for text classification.",
		Flags:       newClientClassifyCommandFlagsFor(app),
		Action:      newClientClassifyCommandActionFor(app),
	}
}

func newClientClassifyCommandFlagsFor(app *BartApp) []cli.Flag {
	return clientutils.Flags(&app.grpcAddress, &app.tlsDisable, &app.output, []cli.Flag{
		cli.StringFlag{
			Name:        "text",
			Destination: &app.requestText,
			Required:    true,
		},
		cli.StringFlag{
			Name:        "text2",
			Destination: &app.requestText2,
			Required:    false,
		},
	})
}

func newClientClassifyCommandActionFor(app *BartApp) func(c *cli.Context) {
	return func(c *cli.Context) {
		clientutils.VerifyFlags(app.output)

		conn := clientutils.OpenConnection(app.grpcAddress, app.tlsDisable)
		client := grpcapi.NewBARTClient(conn)

		resp, err := client.Classify(context.Background(), &grpcapi.ClassifyRequest{
			Text:  app.requestText,
			Text2: app.requestText2,
		})
		if err != nil {
			log.Fatalln(err)
		}

		clientutils.Println(app.output, resp)
	}
}
