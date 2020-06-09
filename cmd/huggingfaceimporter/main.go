// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/docopt/docopt-go"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert"
	"github.com/nlpodyssey/spago/pkg/utils/homedir"
	"log"
	"path"
)

func main() {
	usage := `Usage: hugging_face_importer --model=<name> [--repo=<path>] [--overwrite]

Options:
  --model=<name>   The name of the model to import (as displayed at https://huggingface.co/models).
  --repo=<path>    Directory where to import the model [default: ~/.spago/].
  --overwrite      Overwrite files in case they already exist.
  -h --help        Show this screen.`

	arguments, err := docopt.ParseDoc(usage)
	if err != nil {
		log.Fatal(err)
	}

	overwrite := mustBool(arguments.Bool("--overwrite"))
	model := mustStr(arguments.String("--model"))
	repo := mustStr(arguments.String("--repo"))
	repo, err = homedir.Expand(repo)
	if err != nil {
		log.Fatal(err)
	}

	if err := bert.DownloadHuggingFacePreTrained(repo, model, overwrite); err != nil {
		log.Fatal(err)
	}
	if err := bert.ConvertHuggingFacePreTrained(path.Join(repo, model)); err != nil {
		log.Fatal(err)
	}

	log.Println("Enjoy :)")
}

func mustStr(value string, _ error) string {
	return value
}

func mustBool(value bool, _ error) bool {
	return value
}
