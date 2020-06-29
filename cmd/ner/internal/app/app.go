// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package app

import (
	"github.com/urfave/cli"
)

const (
	programName = "ner-server"
)

var predefinedModels = map[string]string{
	"goflair-en-ner-conll03":      "https://dl.dropboxusercontent.com/s/jgyv568v0nd4ogx/goflair-en-ner-conll03.tar.gz?dl=0",
	"goflair-en-ner-fast-conll03": "https://dl.dropboxusercontent.com/s/9lhh9uom6vh66pg/goflair-en-ner-fast-conll03.tar.gz?dl=0",
}

// NERApp contains everything needed to run the NER client or server.
type NERApp struct {
	*cli.App
	address           string
	grpcAddress       string
	tlsCert           string
	tlsKey            string
	tlsDisable        bool
	output            string
	modelsFolder      string
	modelName         string
	text              string
	mergeEntities     bool
	filterNonEntities bool
}

// NewNERApp returns NerApp objects.
func NewNERApp() *NERApp {
	app := &NERApp{
		App: cli.NewApp(),
	}
	app.Name = programName
	app.Usage = "A demo for named entities recognition."
	app.Commands = []cli.Command{
		newClientCommandFor(app),
		newServerCommandFor(app),
	}
	return app
}
