// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package app

import (
	"context"
	"log"

	"github.com/nlpodyssey/spago/cmd/clientutils"
	"github.com/nlpodyssey/spago/pkg/nlp/sequencelabeler/grpcapi"
	"github.com/urfave/cli"
)

func newClientAnalyzeCommandFor(app *NERApp) cli.Command {
	return cli.Command{
		Name:        "analyze",
		Usage:       "Perform sequence labeling analysis for Named Entity Recognition.",
		UsageText:   programName + " client analyze --text=<text> [--merge-entities] [--filter-non-entities]" + clientutils.UsageText(),
		Description: "Run the " + programName + " client for Named Entity Recognition.",
		Flags:       newClientAnalyzeCommandFlagsFor(app),
		Action:      newClientAnalyzeCommandActionFor(app),
	}
}

func newClientAnalyzeCommandFlagsFor(app *NERApp) []cli.Flag {
	return clientutils.Flags(&app.address, &app.tlsDisable, &app.output, []cli.Flag{
		cli.StringFlag{
			Name:        "text",
			Destination: &app.text,
			Required:    true,
		},
		cli.BoolFlag{
			Name:        "merge-entities",
			Destination: &app.mergeEntities,
		},
		cli.BoolFlag{
			Name:        "filter-non-entities",
			Destination: &app.filterNonEntities,
		},
	})
}

func newClientAnalyzeCommandActionFor(app *NERApp) func(c *cli.Context) {
	return func(c *cli.Context) {
		clientutils.VerifyFlags(app.output)

		conn := clientutils.OpenConnection(app.address, app.tlsDisable)
		client := grpcapi.NewSequenceLabelerClient(conn)

		resp, err := client.Analyze(context.Background(), &grpcapi.AnalyzeRequest{
			Text:              app.text,
			MergeEntities:     app.mergeEntities,
			FilterNotEntities: app.filterNonEntities,
		})

		if err != nil {
			log.Fatalln(err)
		}

		clientutils.Println(app.output, resp)
	}
}
