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

func newClientAnswerCommandFor(app *BertApp) *cli.Command {
	return &cli.Command{
		Name:        "answer",
		Usage:       "Perform question-answering using BERT.",
		Description: "Run the " + programName + " client for question-answering.",
		Flags:       newClientAnswerCommandFlagsFor(app),
		Action:      newClientAnswerCommandActionFor(app),
	}
}

func newClientAnswerCommandFlagsFor(app *BertApp) []cli.Flag {
	return clientutils.Flags(&app.address, &app.tlsDisable, &app.output, []cli.Flag{
		&cli.StringFlag{
			Name:        "passage",
			Destination: &app.passage,
			Required:    true,
		},
		&cli.StringFlag{
			Name:        "question",
			Destination: &app.question,
			Required:    true,
		},
	})
}

func newClientAnswerCommandActionFor(app *BertApp) func(c *cli.Context) error {
	return func(c *cli.Context) error {
		clientutils.VerifyFlags(app.output)

		conn := clientutils.OpenConnection(app.address, app.tlsDisable)
		client := grpcapi.NewBERTClient(conn)

		resp, err := client.Answer(context.Background(), &grpcapi.AnswerRequest{
			Passage:  app.passage,
			Question: app.question,
		})

		if err != nil {
			return err
		}

		clientutils.Println(app.output, resp)

		return nil
	}
}
