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
	"goflair-en-ner-conll03-v0.4":        "https://dl.dropboxusercontent.com/s/nd4hlnzfze2u1ra/goflair-en-ner-conll03-v0.4.tar.gz?dl=0",
	"goflair-en-ner-fast-conll03-v0.4":   "https://dl.dropboxusercontent.com/s/rxf80quo1i64d83/goflair-en-ner-fast-conll03-v0.4.tar.gz?dl=0",
	"goflair-ner-multi-fast":             "https://dl.dropboxusercontent.com/s/u4oa5pv5a27vjsy/goflair-ner-multi-fast.tar.gz?dl=0",
	"goflair-en-ner-ontonotes-fast-v0.4": "https://dl.dropboxusercontent.com/s/jj4lalt52xohmz1/goflair-en-ner-ontonotes-fast-v0.4.tar.gz?dl=0",
	"goflair-fr-ner-wikiner-0.4":         "https://dl.dropboxusercontent.com/s/y7tyykhyowerpdu/goflair-fr-ner-wikiner-0.4.tar.gz?dl=0",
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
	modelFolder       string
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
		newConvertCommandFor(app),
	}
	return app
}
