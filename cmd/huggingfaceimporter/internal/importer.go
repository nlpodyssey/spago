// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/converter"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/huggingfacedownloader"
	"github.com/nlpodyssey/spago/pkg/utils/homedir"
	"github.com/urfave/cli"
	"os"
	"path"
	"strings"
)

const (
	DefaultModelsURL = "https://huggingface.co/models"
	LocalModelsURL   = "local"
	DefaultRepoPath  = "~/.spago/"
	CacheFileName    = "huggingface-co-cache.json"
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
	return []cli.Flag{
		cli.StringFlag{
			Name:        "repo",
			Value:       a.Repo,
			Usage:       "Directory to download the model [default: `DIR`]",
			EnvVar:      "SPAGO_REPO",
			Destination: &a.Repo,
		},
		cli.StringFlag{
			Name:        "model",
			Usage:       "name of the model to load",
			EnvVar:      "SPAGO_MODEL",
			Value:       a.Model,
			Destination: &a.Model,
		},
		cli.StringFlag{
			Name:        "models-url",
			Usage:       "url to lookup models from: `URL`",
			EnvVar:      "SPAGO_MODELS_URL",
			Value:       a.ModelsURL,
			Destination: &a.ModelsURL,
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

	if err := a.ConfigureInteractive(repo); err != nil {
		return err
	}

	writeMsg("Downloading dataset...")

	// make sure the models path exists
	if _, err := os.Stat(repo); os.IsNotExist(err) {
		if err := os.MkdirAll(repo, 0755); err != nil {
			return err
		}
	}

	if a.ModelsURL != LocalModelsURL {
		downloader := huggingfacedownloader.NewDownloader(repo, a.Model, a.Overwrite)
		if err := downloader.Download(); err != nil {
			return err
		}
	}

	return a.convertPreTrainedDataset(repo)
}

func (a *ImporterArgs) convertPreTrainedDataset(repo string) error {
	writeMsg("Configuring/converting dataset...")

	configFilePath := path.Join(repo, a.Model, huggingfacedownloader.ModelConfigFilename)
	config, err := huggingfacedownloader.ReadCommonModelConfig(configFilePath)
	if err != nil {
		return err
	}

	switch config.ModelType {
	case "bart":
		return converter.ConvertHuggingFacePreTrained(path.Join(repo, a.Model))
	case "bert", "electra":
		return bert.ConvertHuggingFacePreTrained(path.Join(repo, a.Model))
	default:
		return fmt.Errorf("unsupported model type: `%s`", config.ModelType)
	}
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
