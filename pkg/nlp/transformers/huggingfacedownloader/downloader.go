// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package huggingfacedownloader

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/utils/httputils"
	"log"
	"os"
	"path"
	"path/filepath"
)

// Downloader provides an easy interface for automatically downloading
// supported pre-trained models from huggingface.co repositories.
type Downloader struct {
	// The local path where all models should be saved.
	modelsPath string
	// The local path which should contain the current model's files.
	modelPath string
	// Hugging Face model name.
	modelName string
	// Full path of the model configuration file.
	configFilePath string
	// Whether existing local files should be overwritten or not.
	canOverwrite bool
}

// NewDownloader creates a new Downloader.
func NewDownloader(modelsPath, modelName string, canOverwrite bool) *Downloader {
	modelPath := filepath.Join(modelsPath, modelName)
	return &Downloader{
		modelsPath:     modelsPath,
		modelPath:      modelPath,
		modelName:      modelName,
		configFilePath: path.Join(modelPath, ModelConfigFilename),
		canOverwrite:   canOverwrite,
	}
}

// Download downloads all the necessary files for the specified model.
func (d *Downloader) Download() error {
	fmt.Printf("Start downloading 🤗 `%s`\n", d.modelName)

	// make sure the models path exists
	if _, err := os.Stat(d.modelsPath); os.IsNotExist(err) {
		return err
	}
	// create the model's directory
	if err := os.MkdirAll(d.modelPath, 0755); err != nil {
		return err
	}

	// fetch configuration file
	if err := d.downloadFile(ModelConfigFilename); err != nil {
		return err
	}

	return d.downloadModelSpecificFiles()
}

func (d *Downloader) downloadModelSpecificFiles() error {
	config, err := ReadCommonModelConfig(d.configFilePath)
	if err != nil {
		return err
	}

	filenames, isSupported := supportedModelsFiles[config.ModelType]
	if !isSupported {
		return fmt.Errorf("unsupported model type: `%s`", config.ModelType)
	}

	for _, filename := range filenames {
		if err := d.downloadFile(filename); err != nil {
			return err
		}
	}
	return nil
}

const (
	// Hugging Face repository URL, in the format:
	// "https://huggingface.co/{model_id}/resolve/{revision}/{filename}"
	huggingFaceCoPrefix = "https://huggingface.co/%s/resolve/%s/%s"
	// Default revision name for fetching models from Hugging Face repository
	defaultRevision = "main"
)

// supportedModelsFiles contains the set of all supported model types as keys,
// mapped with the set of all related files to download.
var supportedModelsFiles = map[string][]string{
	"bart":    {"pytorch_model.bin", "vocab.json", "merges.txt"},
	"bert":    {"pytorch_model.bin", "vocab.txt"},
	"electra": {"pytorch_model.bin", "vocab.txt"},
}

func (d *Downloader) downloadFile(filename string) error {
	filePath := path.Join(d.modelPath, filename)
	if _, err := os.Stat(filePath); !os.IsNotExist(err) && !d.canOverwrite {
		log.Printf("Keeping existing file `%s`\n", filePath)
		return nil
	}

	url := d.bucketURL(filename)
	log.Printf("Fetching file `%s`\n", url)
	if err := httputils.DownloadFile(filePath, url); err != nil {
		return err
	}

	return nil
}

func (d *Downloader) bucketURL(fileName string) string {
	return fmt.Sprintf(huggingFaceCoPrefix, d.modelName, defaultRevision, fileName)
}
