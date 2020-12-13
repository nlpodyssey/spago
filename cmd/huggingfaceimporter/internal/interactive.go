// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

import (
	"github.com/lithammer/fuzzysearch/fuzzy"
	"github.com/manifoldco/promptui"
	"github.com/manifoldco/promptui/list"
	"github.com/pkg/errors"
	"io/ioutil"
	"os"
	"path"
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

// ConfigureInteractive uses the CLI to configure.
func (a *ImporterArgs) ConfigureInteractive(repo string) error {
	if a.Model == "" {
		models := make([]string, len(SuggestedModels)+2)
		copy(models, SuggestedModels)
		otherStr := "Enter name"
		models[len(models)-1] = otherStr
		searchStr := "Search"
		models[len(models)-2] = searchStr
		pr := &promptui.Select{
			Label: "Model from " + a.ModelsURL,
			Items: models,
		}
		var err error
		modelsURL := a.ModelsURL
		_, a.Model, err = pr.Run()
		if err != nil {
			return err
		}
		if a.Model == otherStr {
			a.Model, err = (&promptui.Prompt{Label: "Model"}).Run()
			if err != nil {
				return err
			}
		} else if a.Model == searchStr {
			// Check if data already is cached.
			cacheFilePath := path.Join(repo, CacheFileName)
			var dataJSON string
			if _, err := os.Stat(cacheFilePath); err == nil {
				dataBin, err := ioutil.ReadFile(cacheFilePath)
				if err != nil {
					writeMsg("Could not read cache file, skipping: " + err.Error())
				}
				dataJSON = string(dataBin)
			}
			if len(dataJSON) == 0 {
				writeMsg("Loading data from " + modelsURL)
				dataJSON, err = LookupFromHuggingFace("")
				if err != nil {
					return errors.Wrap(err, "load data from "+modelsURL)
				}
			}
			if len(dataJSON) == 0 {
				return errors.New("fetch returned no data from " + modelsURL)
			}
			// parse
			srData, err := ParseSearchResults([]byte(dataJSON))
			if err != nil {
				return errors.Wrap(err, "parse search results data")
			}
			// write cache
			if err := ioutil.WriteFile(cacheFilePath, []byte(dataJSON), 0644); err != nil {
				writeMsg("Unable to write cache file: " + err.Error())
			}

			// convert to IDs
			ids := make([]string, len(srData))
			for i := 0; i < len(ids); i++ {
				ids[i] = srData[i].ModelID
			}

			_, a.Model, err = (&promptui.Select{
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
	if a.Model == "" {
		return errors.New("no model selected")
	}
	return nil
}
