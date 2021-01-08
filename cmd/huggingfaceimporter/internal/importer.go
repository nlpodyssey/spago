// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/huggingface"
	"github.com/nlpodyssey/spago/pkg/utils/homedir"
	"github.com/urfave/cli"
	"log"
	"os"
	"os/user"
	"path"
	"path/filepath"
	"strings"
)

const (
	// DefaultModelsURL can be used as value for ImporterArgs.ModelsURL to
	// download models from Hugging Face website.
	DefaultModelsURL = "https://huggingface.co/models"
	// DefaultRepoPath is the default base path for all locally stored models.
	DefaultRepoPath = "~/.spago/"
	// CacheFileName is the name of the JSON file where the Hugging Face Importer
	// should store cache data.
	CacheFileName = "huggingface-co-cache.json"
)

// ImporterArgs contain args for the import command (default)
type ImporterArgs struct {
	Repo      string
	Model     string
	ModelsURL string
	Overwrite bool
}

// NewImporterArgs builds args object.
func NewImporterArgs(repo, model, modelsURL string, overwrite bool) *ImporterArgs {
	return &ImporterArgs{
		Repo:      repo,
		Model:     model,
		ModelsURL: modelsURL,
		Overwrite: overwrite,
	}
}

// NewDefaultImporterArgs builds the args with defaults.
func NewDefaultImporterArgs() *ImporterArgs {
	return NewImporterArgs(DefaultRepoPath, "", DefaultModelsURL, false)
}

// BuildFlags builds the flags for the args.
func (a *ImporterArgs) BuildFlags() []cli.Flag {
	usr, err := user.Current()
	if err != nil {
		log.Fatal(err)
	}

	return []cli.Flag{
		cli.StringFlag{
			Name:        "repo",
			Value:       path.Join(usr.HomeDir, ".spago"),
			Usage:       "Directory to download the model [default: `DIR`]",
			EnvVar:      "SPAGO_REPO",
			Destination: &a.Repo,
		},
		cli.StringFlag{
			Name:        "model",
			Usage:       "name of the model to convert",
			EnvVar:      "SPAGO_MODEL",
			Value:       a.Model,
			Destination: &a.Model,
		},
		cli.BoolFlag{
			Name:        "overwrite",
			Usage:       "overwrite files if they exist already",
			EnvVar:      "SPAGO_OVERWRITE",
			Destination: &a.Overwrite,
		},
	}
}

// RunImporter runs the importer.
func (a *ImporterArgs) RunImporter() error {
	repo, err := homedir.Expand(a.Repo)
	if err != nil {
		return err
	}

	// Run interactive model selection if a model is not already set.
	if a.Model == "" {
		if err := a.ConfigureInteractive(repo); err != nil {
			return err
		}
	}

	// make sure the models path exists
	if _, err := os.Stat(a.Repo); os.IsNotExist(err) {
		if err := os.MkdirAll(a.Repo, 0755); err != nil {
			return err
		}
	}

	modelPath := filepath.Join(a.Repo, a.Model)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) || a.Overwrite {
		fmt.Printf("Pulling `%s` from Hugging Face models hub...\n", a.Model)
		err = huggingface.NewDownloader(a.Repo, a.Model, true).Download()
		if err != nil {
			return err
		}
	}

	fmt.Printf("Converting `%s` model...\n", a.Model)
	return huggingface.NewConverter(a.Repo, a.Model).Convert()
}

// RunImporterCli runs the importer from the command line.
func (a *ImporterArgs) RunImporterCli(_ *cli.Context) error {
	return a.RunImporter()
}

func writeMsg(m string) {
	_, _ = os.Stderr.WriteString(m)
	if !strings.HasSuffix(m, "\n") {
		_, _ = os.Stderr.WriteString("\n")
	}
}
