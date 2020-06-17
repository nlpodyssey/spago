package main

import (
	"io/ioutil"
	"os"
	"path"
	"strings"

	"github.com/lithammer/fuzzysearch/fuzzy"
	"github.com/manifoldco/promptui"
	"github.com/manifoldco/promptui/list"
	"github.com/pkg/errors"
)

func writeMsg(m string) {
	os.Stderr.WriteString(m)
	if !strings.HasSuffix(m, "\n") {
		os.Stderr.WriteString("\n")
	}
}

// ConfigureInteractive uses the CLI to configure.
func (a *ImporterArgs) ConfigureInteractive(repo string) error {
	if a.Model == "" {
		models := SuggestedModels
		mmodels := make([]string, len(models)+2)
		copy(mmodels, models)
		otherStr := "Enter name"
		mmodels[len(mmodels)-1] = otherStr
		searchStr := "Search the directory"
		mmodels[len(mmodels)-2] = searchStr
		pr := &promptui.Select{
			Label: "Model from " + a.ModelsURL,
			Items: mmodels,
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
			cacheFilePath := path.Join(repo, CachefileName)
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
