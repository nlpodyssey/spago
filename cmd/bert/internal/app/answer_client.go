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

func newClientAnswerCommandFor(app *BertApp) cli.Command {
	return cli.Command{
		Name:        "answer",
		Usage:       "Perform question-answering using BERT.",
		UsageText:   programName + " client answer --passage=<passage> --question=<question> [--address=<address>] [--tls-disable]",
		Description: "Run the " + programName + " client for question-answering.",
		Flags:       newClientAnswerCommandFlagsFor(app),
		Action:      newClientAnswerCommandActionFor(app),
	}
}

func newClientAnswerCommandFlagsFor(app *BertApp) []cli.Flag {
	return []cli.Flag{
		cli.StringFlag{
			Name:        "address",
			Value:       "127.0.0.1:1976",
			Destination: &app.address,
		},
		cli.StringFlag{
			Name:        "passage",
			Destination: &app.passage,
			Required:    true,
		},
		cli.StringFlag{
			Name:        "question",
			Destination: &app.question,
			Required:    true,
		},
		cli.BoolFlag{
			Name:        "tls-disable ",
			Usage:       "Specifies that TLS is disabled.",
			Destination: &app.tlsDisable,
		},
	}
}

func newClientAnswerCommandActionFor(app *BertApp) func(c *cli.Context) {
	return func(c *cli.Context) {
		conn := grpcutils.OpenClientConnection(app.address, app.tlsDisable)
		cli := grpcapi.NewBERTClient(conn)

		resp, err := cli.Answer(context.Background(), &grpcapi.AnswerRequest{
			Passage:  app.passage,
			Question: app.question,
		})

		if err != nil {
			log.Fatalln(err)
		}

		fmt.Println(resp)
	}
}
