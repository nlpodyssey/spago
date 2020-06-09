// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/nlpodyssey/spago/pkg/utils/httputils"
	"log"
	"net/url"
	"os"
	"path"
	"path/filepath"
)

// TODO: This code needs to be refactored. Pull requests are welcome!

const (
	storageURL               = "https://s3.amazonaws.com/models.huggingface.co/bert/"
	defaultConfigurationFile = "config.json"
	defaultModelFile         = "pytorch_model.bin"
	defaultVocabularyFile    = "vocab.txt"
)

var supportedModelsType = map[string]bool{"bert": true, "electra": true}

var legacyModels = map[string]bool{
	"bert-base-uncased":                                     true,
	"bert-large-uncased":                                    true,
	"bert-base-cased":                                       true,
	"bert-large-cased":                                      true,
	"bert-base-multilingual-uncased":                        true,
	"bert-base-multilingual-cased":                          true,
	"bert-base-chinese":                                     true,
	"bert-base-german-cased":                                true,
	"bert-large-uncased-whole-word-masking":                 true,
	"bert-large-cased-whole-word-masking":                   true,
	"bert-large-uncased-whole-word-masking-finetuned-squad": true,
	"bert-large-cased-whole-word-masking-finetuned-squad":   true,
	"bert-base-cased-finetuned-mrpc":                        true,
	"bert-base-german-dbmdz-cased":                          true,
	"bert-base-german-dbmdz-uncased":                        true,
}

func DownloadHuggingFacePreTrained(modelsPath, modelName string, overwrite bool) error {
	handler := &huggingFacePreTrainedDownloader{
		ModelsPath: modelsPath,
		ModelName:  modelName,
		Overwrite:  overwrite,
	}
	return handler.download()
}

type huggingFacePreTrainedDownloader struct {
	ModelsPath string
	ModelName  string
	modelPath  string
	Overwrite  bool
}

func (d *huggingFacePreTrainedDownloader) download() error {
	fmt.Printf("Start downloading 🤗 `%s`\n", d.ModelName)
	// make sure the models path exists
	if _, err := os.Stat(d.ModelsPath); os.IsNotExist(err) {
		return err
	}
	// create the model's directory
	d.modelPath = filepath.Join(d.ModelsPath, d.ModelName)
	if err := os.MkdirAll(d.modelPath, 0755); err != nil {
		return err
	}
	// fetching resources
	if err := d.fetchConfig(); err != nil {
		return err
	}
	if err := d.fetchVocabulary(); err != nil {
		return err
	}
	if err := d.fetchModelWeights(); err != nil {
		return err
	}
	return nil
}

func (d *huggingFacePreTrainedDownloader) fetchConfig() error {
	configURL := buildConfigURL(d.ModelName)
	configPath := path.Join(d.modelPath, defaultConfigurationFile)
	if _, err := os.Stat(configPath); os.IsNotExist(err) || d.Overwrite {
		log.Printf("Fetch the model configuration from `%s`\n", configURL)
		if err := httputils.DownloadFile(configPath, configURL); err != nil {
			return err
		}
	} else {
		log.Printf("Keep existing configuration `%s`\n", configPath)
	}
	if err := checkConfig(configPath); err != nil {
		return err
	}
	return nil
}

func (d *huggingFacePreTrainedDownloader) fetchVocabulary() error {
	vocabURL := buildVocabURL(d.ModelName)
	vocabPath := path.Join(d.modelPath, defaultVocabularyFile)
	if _, err := os.Stat(vocabPath); os.IsNotExist(err) || d.Overwrite {
		log.Printf("Fetch the model vocabulary from `%s`\n", vocabURL)
		if err := httputils.DownloadFile(vocabPath, vocabURL); err != nil {
			return err
		}
	} else {
		log.Printf("Keep existing vocabulary `%s`\n", vocabPath)
	}
	return nil
}

func (d *huggingFacePreTrainedDownloader) fetchModelWeights() error {
	modelURL := buildModelURL(d.ModelName)
	modelPath := path.Join(d.modelPath, defaultModelFile)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) || d.Overwrite {
		log.Printf("Fetch the model weights from `%s` (it might take a while...)\n", modelURL)
		if err := httputils.DownloadFile(modelPath, modelURL); err != nil {
			return err
		}
	} else {
		log.Printf("Keep existing model weights `%s`\n", modelPath)
	}
	return nil
}

func buildConfigURL(modelName string) string {
	u, _ := url.Parse(storageURL) // TODO: error handling
	if _, exist := legacyModels[modelName]; exist {
		u.Path = path.Join(u.Path, modelName)
		u.Path = fmt.Sprintf("%s-%s", u.Path, defaultConfigurationFile)
	} else {
		u.Path = path.Join(u.Path, modelName, defaultConfigurationFile)
	}
	return u.String()
}

func buildVocabURL(modelName string) string {
	u, _ := url.Parse(storageURL) // TODO: error handling
	if _, exist := legacyModels[modelName]; exist {
		u.Path = path.Join(u.Path, modelName)
		u.Path = fmt.Sprintf("%s-%s", u.Path, defaultVocabularyFile)
	} else {
		u.Path = path.Join(u.Path, modelName, defaultVocabularyFile)
	}
	return u.String()
}

func buildModelURL(modelName string) string {
	u, _ := url.Parse(storageURL) // TODO: error handling
	if _, exist := legacyModels[modelName]; exist {
		u.Path = path.Join(u.Path, modelName)
		u.Path = fmt.Sprintf("%s-%s", u.Path, defaultModelFile)
	} else {
		u.Path = path.Join(u.Path, modelName, defaultModelFile)
	}
	return u.String()
}

func checkConfig(filepath string) error {
	var result map[string]interface{}
	configFile, err := os.Open(filepath)
	if err != nil {
		return errors.New(fmt.Sprintf("config file not found; %s", err.Error()))
	}
	defer configFile.Close()
	err = json.NewDecoder(configFile).Decode(&result)
	if err != nil {
		return errors.New(fmt.Sprintf("invalid config file; %s", err.Error()))
	}
	modelType, exist := result["model_type"]
	if !exist {
		return errors.New("expected key model_type")
	}
	if _, supported := supportedModelsType[modelType.(string)]; !supported {
		return errors.New(fmt.Sprintf("unsupported model type `%s`", modelType))
	}
	return nil
}
