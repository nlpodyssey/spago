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
	"goflair-en-ner-conll03-v0.4":          "https://dl.dropboxusercontent.com/s/fes2dvjiczrwp8m/goflair-en-ner-conll03-v0.4.tar.gz?dl=0",
	"goflair-en-ner-fast-conll03-v0.4":     "https://dl.dropboxusercontent.com/s/w3dlvc75vao8h32/goflair-en-ner-fast-conll03-v0.4.tar.gz?dl=0",
	"goflair-ner-multi-fast":               "https://dl.dropboxusercontent.com/s/i980dpfowaecp4t/goflair-ner-multi-fast.tar.gz?dl=0",
	"goflair-en-ner-ontonotes-fast-v0.4":   "https://dl.dropboxusercontent.com/s/ycdcm4h3uvogfah/goflair-en-ner-ontonotes-fast-v0.4.tar.gz?dl=0",
	"goflair-fr-ner-wikiner-0.4":           "https://dl.dropboxusercontent.com/s/njvs8d9iunrewxy/goflair-fr-ner-wikiner-0.4.tar.gz?dl=0",
	"goflair-en-chunk-conll2000-fast-v0.4": "https://dl.dropboxusercontent.com/s/0b3wl9t2t1sczzv/goflair-en-chunk-conll2000-fast-v0.4.tar.gz?dl=0",
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
	repo              string
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
	app.HelpName = programName
	app.Usage = "A demo for named entities recognition."
	app.Commands = []cli.Command{
		newClientCommandFor(app),
		newServerCommandFor(app),
		newConvertCommandFor(app),
	}
	return app
}
