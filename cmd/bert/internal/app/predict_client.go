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

func newClientPredictCommandFor(app *BertApp) cli.Command {
	return cli.Command{
		Name:        "predict",
		Usage:       "Perform a prediction based on a trained Masked Language Model (MLM).",
		UsageText:   programName + " client predict --text=<value> [--address=<address>] [--tls-disable]",
		Description: "Run the " + programName + " client for prediction.",
		Flags:       newClientPredictCommandFlagsFor(app),
		Action:      newClientPredictCommandActionFor(app),
	}
}

func newClientPredictCommandFlagsFor(app *BertApp) []cli.Flag {
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

func newClientPredictCommandActionFor(app *BertApp) func(c *cli.Context) {
	return func(c *cli.Context) {
		conn := grpcutils.OpenClientConnection(app.address, app.tlsDisable)
		cli := grpcapi.NewBERTClient(conn)

		resp, err := cli.Predict(context.Background(), &grpcapi.PredictRequest{
			Text: app.requestText,
		})

		if err != nil {
			log.Fatalln(err)
		}

		fmt.Println(resp)
	}
}
