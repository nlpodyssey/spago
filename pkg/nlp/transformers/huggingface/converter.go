// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package huggingface

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/converter"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert"
	"path"
	"path/filepath"
)

// Converter provides an easy interface for automatically converting
// supported pre-trained models from huggingface.co repositories.
type Converter struct {
	// The local path where all models should be saved.
	modelsPath string
	// The local path which should contain the current model's files.
	modelPath string
	// Hugging Face model name.
	modelName string
	// Full path of the model configuration file.
	configFilename string
}

// NewConverter creates a new Converter.
func NewConverter(modelsPath, modelName string) *Converter {
	modelPath := filepath.Join(modelsPath, modelName)
	return &Converter{
		modelsPath:     modelsPath,
		modelPath:      modelPath,
		modelName:      modelName,
		configFilename: path.Join(modelPath, ModelConfigFilename),
	}
}

// Convert converts the pickle-serialized model to spaGO.
func (c *Converter) Convert() error {
	config, err := ReadCommonModelConfig(c.configFilename)
	if err != nil {
		return err
	}

	switch config.ModelType {
	case "bart":
		return converter.ConvertHuggingFacePreTrained(c.modelPath)
	case "bert", "electra":
		return bert.ConvertHuggingFacePreTrained(c.modelPath)
	case "":
		fmt.Println("model type empty; assuming it is BERT.")
		return bert.ConvertHuggingFacePreTrained(c.modelPath)
	default:
		return fmt.Errorf("unsupported model type: `%s`", config.ModelType)
	}
}
