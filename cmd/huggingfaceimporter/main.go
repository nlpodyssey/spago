// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"io/ioutil"
	"os"
	"path"
	"strings"

	fuzzy "github.com/lithammer/fuzzysearch/fuzzy"
	"github.com/manifoldco/promptui"
	"github.com/manifoldco/promptui/list"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert"
	"github.com/nlpodyssey/spago/pkg/utils/homedir"
	"github.com/pkg/errors"
	"github.com/urfave/cli"
)

func writeMsg(m string) {
	os.Stderr.WriteString(m)
	if !strings.HasSuffix(m, "\n") {
		os.Stderr.WriteString("\n")
	}
}

var (
	repo      string
	model     string
	overwrite bool
)

const modelsURL = "https://huggingface.co/models"

const cacheName = "huggingface-co-cache.json"

// Models are excellent models to start with.
var models = []string{
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
	app := cli.NewApp()
	app.Name = "huggingfaceimporter"
	app.Usage = "import data to a target path"
	app.HideVersion = true
	app.Action = runAutoDemo
	app.Flags = []cli.Flag{
		cli.StringFlag{
			Name:        "repo",
			Value:       "~/.spago/",
			Usage:       "Directory to download the model [default: `DIR`]",
			EnvVar:      "SPAGO_REPO",
			Destination: &repo,
		},
		cli.StringFlag{
			Name:        "model",
			Usage:       "name of the model from " + modelsURL,
			EnvVar:      "SPAGO_MODEL",
			Destination: &model,
		},
		cli.BoolFlag{
			Name:        "overwrite",
			Usage:       "overwrite files if they exist already",
			EnvVar:      "SPAGO_OVERWRITE",
			Destination: &overwrite,
		},
	}

	if err := app.Run(os.Args); err != nil {
		writeMsg(err.Error())
		os.Exit(1)
	}
}

func runAutoDemo(_ *cli.Context) error {
	var err error
	repo, err = homedir.Expand(repo)
	if err != nil {
		return err
	}

	if err := configureInteractive(); err != nil {
		return err
	}

	writeMsg("Downloading dataset...")

	// make sure the models path exists
	if _, err := os.Stat(repo); os.IsNotExist(err) {
		if err := os.MkdirAll(repo, 0755); err != nil {
			return err
		}
	}

	if err := bert.DownloadHuggingFacePreTrained(repo, model, overwrite); err != nil {
		return err
	}

	writeMsg("Configuring/converting dataset...")
	if err := bert.ConvertHuggingFacePreTrained(path.Join(repo, model)); err != nil {
		return err
	}

	return nil
}

func configureInteractive() error {
	if model == "" {
		mmodels := make([]string, len(models)+2)
		copy(mmodels, models)
		otherStr := "Enter name"
		mmodels[len(mmodels)-1] = otherStr
		searchStr := "Search the directory"
		mmodels[len(mmodels)-2] = searchStr
		pr := &promptui.Select{
			Label: "Model from " + modelsURL,
			Items: mmodels,
		}
		var err error
		_, model, err = pr.Run()
		if err != nil {
			return err
		}
		if model == otherStr {
			model, err = (&promptui.Prompt{Label: "Model"}).Run()
			if err != nil {
				return err
			}
		} else if model == searchStr {
			// Check if data already is cached.
			cacheFilePath := path.Join(repo, cacheName)
			var dataJson string
			if _, err := os.Stat(cacheFilePath); err == nil {
				dataBin, err := ioutil.ReadFile(cacheFilePath)
				if err != nil {
					writeMsg("Could not read cache file, skipping: " + err.Error())
				}
				dataJson = string(dataBin)
			}
			if len(dataJson) == 0 {
				writeMsg("Loading data from " + modelsURL)
				dataJson, err = LookupFromHuggingFace("")
				if err != nil {
					return errors.Wrap(err, "load data from "+modelsURL)
				}
			}
			if len(dataJson) == 0 {
				return errors.New("fetch returned no data from " + modelsURL)
			}
			// parse
			srData, err := ParseSearchResults([]byte(dataJson))
			if err != nil {
				return errors.Wrap(err, "parse search results data")
			}
			// write cache
			if err := ioutil.WriteFile(cacheFilePath, []byte(dataJson), 0644); err != nil {
				writeMsg("Unable to write cache file: " + err.Error())
			}

			// convert to IDs
			ids := make([]string, len(srData))
			for i := 0; i < len(ids); i++ {
				ids[i] = srData[i].ModelID
			}

			_, model, err = (&promptui.Select{
				Label:             "Model search",
				StartInSearchMode: true,
				Items:             ids,
				// Called on each items of the select and should return a
				// boolean for whether or not the item fits the searched term.
				Searcher: list.Searcher(func(input string, index int) bool {
					return fuzzy.Match(input, ids[index])
				}),
			}).Run()
			if err != nil {
				return err
			}
		}
	}
	if model == "" {
		return errors.New("no model selected")
	}
	return nil
}
