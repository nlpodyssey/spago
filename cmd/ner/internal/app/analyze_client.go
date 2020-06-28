// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package app

import (
	"context"
	"fmt"
	"log"

	"github.com/nlpodyssey/spago/pkg/nlp/sequencelabeler/grpcapi"
	"github.com/nlpodyssey/spago/pkg/utils/grpcutils"
	"github.com/urfave/cli"
)

func newClientAnalyzeCommandFor(app *NERApp) cli.Command {
	return cli.Command{
		Name:        "analyze",
		Usage:       "Perform sequence labeling analysis for Named Entity Recognition.",
		UsageText:   programName + " client analyze --text=<text> [--merge-entities] [--filter-non-entities] [--address=<address>] [--tls-disable]",
		Description: "Run the " + programName + " client for Named Entity Recognition.",
		Flags:       newClientAnalyzeCommandFlagsFor(app),
		Action:      newClientAnalyzeCommandActionFor(app),
	}
}

func newClientAnalyzeCommandFlagsFor(app *NERApp) []cli.Flag {
	return []cli.Flag{
		cli.StringFlag{
			Name:        "address",
			Value:       "127.0.0.1:1976",
			Destination: &app.address,
		},
		cli.StringFlag{
			Name:        "text",
			Destination: &app.text,
		},
		cli.BoolFlag{
			Name:        "merge-entities",
			Destination: &app.mergeEntities,
		},
		cli.BoolFlag{
			Name:        "filter-non-entities",
			Destination: &app.filterNonEntities,
		},
		cli.BoolFlag{
			Name:        "tls-disable ",
			Usage:       "Specifies that TLS is disabled.",
			Destination: &app.tlsDisable,
		},
	}
}

func newClientAnalyzeCommandActionFor(app *NERApp) func(c *cli.Context) {
	return func(c *cli.Context) {
		conn := grpcutils.OpenClientConnection(app.address, app.tlsDisable)
		cli := grpcapi.NewSequenceLabelerClient(conn)

		resp, err := cli.Analyze(context.Background(), &grpcapi.AnalyzeRequest{
			Text:              app.text,
			MergeEntities:     app.mergeEntities,
			FilterNotEntities: app.filterNonEntities,
		})

		if err != nil {
			log.Fatalln(err)
		}

		fmt.Println(resp)
	}
}
