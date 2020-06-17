// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/urfave/cli"
	"os"
)

// SuggestedModels are excellent models to start with.
var SuggestedModels = []string{
	// suggested models for question-answering
	"deepset/bert-base-cased-squad2",
	// suggested models for real/fake token discrimination
	"google/electra-base-discriminator",
	// suggested models for masked tokens prediction
	"bert-base-cased",
	"bert-base-multilingual-cased",
	"bert-base-german-cased",
}

func main() {
	importerArgs := NewDefaultImporterArgs()
	importerFlags := importerArgs.BuildFlags()

	app := cli.NewApp()
	app.Name = "huggingfaceimporter"
	app.Usage = "import data to a target path"
	app.HideVersion = true
	app.Action = importerArgs.RunImporterCli
	app.Flags = importerFlags

	if err := app.Run(os.Args); err != nil {
		writeMsg(err.Error())
		os.Exit(1)
	}
}
