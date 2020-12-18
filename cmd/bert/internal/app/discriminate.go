// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package app

import (
	"context"
	"log"

	"github.com/nlpodyssey/spago/cmd/clientutils"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert/grpcapi"
	"github.com/urfave/cli"
)

func newClientDiscriminateCommandFor(app *BertApp) cli.Command {
	return cli.Command{
		Name:        "discriminate",
		Usage:       "Perform linear discriminate analysis using BERT.",
		UsageText:   programName + " client discriminate --text=<value>" + clientutils.UsageText(),
		Description: "Run the " + programName + " client for linear discriminate analysis.",
		Flags:       newClientDiscriminateCommandFlagsFor(app),
		Action:      newClientDiscriminateCommandActionFor(app),
	}
}

func newClientDiscriminateCommandFlagsFor(app *BertApp) []cli.Flag {
	return clientutils.Flags(&app.address, &app.tlsDisable, &app.output, []cli.Flag{
		cli.StringFlag{
			Name:        "text",
			Destination: &app.requestText,
			Required:    true,
		},
	})
}

func newClientDiscriminateCommandActionFor(app *BertApp) func(c *cli.Context) {
	return func(c *cli.Context) {
		clientutils.VerifyFlags(app.output)

		conn := clientutils.OpenConnection(app.address, app.tlsDisable)
		client := grpcapi.NewBERTClient(conn)

		resp, err := client.Discriminate(context.Background(), &grpcapi.DiscriminateRequest{
			Text: app.requestText,
		})

		if err != nil {
			log.Fatalln(err)
		}

		clientutils.Println(app.output, resp)
	}
}
