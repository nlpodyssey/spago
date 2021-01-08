// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/nlpodyssey/spago/cmd/huggingfaceimporter/internal"
	"github.com/urfave/cli"
	"os"
	"strings"
)

func main() {
	importerArgs := internal.NewDefaultImporterArgs()
	importerFlags := importerArgs.BuildFlags()

	app := cli.NewApp()
	app.Name = "huggingface-importer"
	app.Usage = "Convert a pickle-serialized Hugging Face model to spaGO"
	app.HideVersion = true
	app.Action = importerArgs.RunImporterCli
	app.Flags = importerFlags

	if err := app.Run(os.Args); err != nil {
		writeMsg(err.Error())
		os.Exit(1)
	}
}

func writeMsg(m string) {
	_, _ = os.Stderr.WriteString(m)
	if !strings.HasSuffix(m, "\n") {
		_, _ = os.Stderr.WriteString("\n")
	}
}
