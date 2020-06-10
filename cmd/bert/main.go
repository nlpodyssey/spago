// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"github.com/docopt/docopt-go"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert"
	"log"
	"os"
	"strconv"
)

func main() {
	usage := `Usage: bert_server --model=<path> --port=<port>

Options:
  -m --model=<path>   The path of the model to load.
  -p --port=<port>    The API port. 
  -h --help           Show this screen.`

	arguments, err := docopt.ParseDoc(usage)
	if err != nil {
		log.Fatal(err)
	}

	port, err := strconv.Atoi(mustStr(arguments.String("--port")))
	if err != nil {
		log.Fatal(err)
	}

	modelPath := mustStr(arguments.String("--model"))
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		log.Fatal(err)
	}

	model, err := bert.LoadModel(modelPath)
	if err != nil {
		log.Fatal(fmt.Sprintf("error during model loading (%s)", err.Error()))
	}
	fmt.Printf("Config: %+v\n", model.Config)

	fmt.Println(fmt.Sprintf("Start server on port %d.", port))
	server := bert.NewServer(model, port)
	server.Start()
}

func mustStr(value string, _ error) string {
	return value
}
