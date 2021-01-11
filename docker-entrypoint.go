// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	bartapp "github.com/nlpodyssey/spago/cmd/bart/app"
	bertapp "github.com/nlpodyssey/spago/cmd/bert/app"
	huggingfaceimporterapp "github.com/nlpodyssey/spago/cmd/huggingfaceimporter/app"
	nerapp "github.com/nlpodyssey/spago/cmd/ner/app"
	"log"
	"os"
)

const help = `
spaGO is a self-contained ML/NLP library written in Go.

Usage:
    <command> [arguments]

The commands are:

    bert-server             gRPC/HTTP server for BERT
    bart-server             gRPC/HTTP server for BART
    huggingface-importer    Hugging Face model importing
    ner-server              gRPC/HTTP server for Sequence Labeling
    help                    print this help text and exit

See README.md for more information about run the demo servers using Docker.
`

// The script docker-entrypoint.sh wraps access to the cmd programs.
func main() {
	logger := log.New(os.Stderr, "", 0)

	if len(os.Args) == 1 {
		logger.Fatalf("Command missing.\n%s", help)
	}

	command := os.Args[1]
	args := os.Args[1:]

	var err error

	switch command {
	case "help":
		fmt.Print(help)
	case "bert-server":
		err = bertapp.NewBertApp().Run(args)
	case "bart-server":
		err = bartapp.NewBartApp().Run(args)
	case "huggingface-importer":
		err = huggingfaceimporterapp.New().Run(args)
	case "ner-server":
		err = nerapp.NewNERApp().Run(args)
	default:
		logger.Fatalf("Invalid command.\n%s", help)
	}

	if err != nil {
		logger.Fatal(err)
	}
}
