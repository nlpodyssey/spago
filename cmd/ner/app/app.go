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
	"goflair-en-ner-conll03-v0.4":          "https://dl.dropboxusercontent.com/s/41fk31b684mz6v8/goflair-en-ner-conll03-v0.4.tar.gz?dl=0",
	"goflair-en-ner-fast-conll03-v0.4":     "https://dl.dropboxusercontent.com/s/5zlyw1e6gpm94oo/goflair-en-ner-fast-conll03-v0.4.tar.gz?dl=0",
	"goflair-ner-multi-fast":               "https://dl.dropboxusercontent.com/s/k5kvaf23oorkgv3/goflair-ner-multi-fast.tar.gz?dl=0",
	"goflair-en-ner-ontonotes-fast-v0.4":   "https://dl.dropboxusercontent.com/s/v2r2cldnz9rh6ya/goflair-en-ner-ontonotes-fast-v0.4.tar.gz?dl=0",
	"goflair-fr-ner-wikiner-0.4":           "https://dl.dropboxusercontent.com/s/y3gv1gz4ckmj8bn/goflair-fr-ner-wikiner-0.4.tar.gz?dl=0",
	"goflair-en-chunk-conll2000-fast-v0.4": "https://dl.dropboxusercontent.com/s/un8wxcsqweidg1x/goflair-en-chunk-conll2000-fast-v0.4.tar.gz?dl=0",
}

// NERApp contains everything needed to run the NER client or server.
type NERApp struct {
	*cli.App
	address               string
	grpcAddress           string
	tlsCert               string
	tlsKey                string
	tlsDisable            bool
	output                string
	repo                  string
	modelFolder           string
	modelName             string
	text                  string
	mergeEntities         bool
	filterNonEntities     bool
	serverTimeoutSeconds  int
	serverMaxRequestBytes int
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
