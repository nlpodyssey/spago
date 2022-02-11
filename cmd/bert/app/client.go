// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package app

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/urfave/cli/v2"
)

func newClientCommandFor[T mat.DType](app *BertApp) *cli.Command {
	return &cli.Command{
		Name:  "client",
		Usage: "Run the " + programName + " client.",
		Subcommands: []*cli.Command{
			newClientAnswerCommandFor(app),
			newClientDiscriminateCommandFor(app),
			newClientPredictCommandFor(app),
			newClientEncodeCommandFor(app),
			newClientSimilarityCommandFor[T](app),
			newClientClassifyCommandFor(app),
		},
	}
}
