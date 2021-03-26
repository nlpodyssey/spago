// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package app

import (
	"context"

	"github.com/nlpodyssey/spago/cmd/clientutils"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert/grpcapi"
	"github.com/urfave/cli/v2"
)

func newClientPredictCommandFor(app *BertApp) *cli.Command {
	return &cli.Command{
		Name:        "predict",
		Usage:       "Perform a prediction based on a trained Masked Language Model (MLM).",
		Description: "Run the " + programName + " client for prediction.",
		Flags:       newClientPredictCommandFlagsFor(app),
		Action:      newClientPredictCommandActionFor(app),
	}
}

func newClientPredictCommandFlagsFor(app *BertApp) []cli.Flag {
	return clientutils.Flags(&app.address, &app.tlsDisable, &app.output, []cli.Flag{
		&cli.StringFlag{
			Name:        "text",
			Destination: &app.requestText,
			Required:    true,
		},
	})
}

func newClientPredictCommandActionFor(app *BertApp) func(c *cli.Context) error {
	return func(c *cli.Context) error {
		clientutils.VerifyFlags(app.output)

		conn := clientutils.OpenConnection(app.address, app.tlsDisable)
		client := grpcapi.NewBERTClient(conn)

		resp, err := client.Predict(context.Background(), &grpcapi.PredictRequest{
			Text: app.requestText,
		})

		if err != nil {
			return err
		}

		clientutils.Println(app.output, resp)

		return nil
	}
}
