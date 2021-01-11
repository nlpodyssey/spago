// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package app

import (
	"github.com/nlpodyssey/spago/cmd/huggingfaceimporter/internal"
	"github.com/urfave/cli"
)

const (
	programName = "huggingface-importer"
)

// New returns a new CLI App for importing pickle-serialized Hugging Face models.
func New() *cli.App {
	importerArgs := internal.NewDefaultImporterArgs()
	importerFlags := importerArgs.BuildFlags()

	app := cli.NewApp()
	app.Name = programName
	app.HelpName = programName
	app.Usage = "Convert a pickle-serialized Hugging Face model to spaGO"
	app.HideVersion = true
	app.Action = importerArgs.RunImporterCli
	app.Flags = importerFlags
	return app
}
