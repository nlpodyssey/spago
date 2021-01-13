// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package huggingface

import (
	"encoding/json"
	"fmt"
	"os"
)

// ModelConfigFilename is the default configuration filename for all Hugging
// Face pre-trained models.
const ModelConfigFilename = "config.json"

// CommonModelConfig provides the bare minimum set of model configuration
// properties which are shared among models of different types.
//
// This is useful when you need to perform different actions depending on
// the value of certain basic common settings.
//
// For example, the Downloader uses this information to roughly validate
// the JSON configuration data and to decide how to proceed with further files
// to download.
type CommonModelConfig struct {
	ModelType string `json:"model_type"`
}

// ReadCommonModelConfig parses the given JSON config file, returning a new
// CommonModelConfig value.
func ReadCommonModelConfig(filename string) (cmc *CommonModelConfig, err error) {
	configFile, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("JSON config file not found: %w", err)
	}
	defer func() {
		if e := configFile.Close(); e != nil && err == nil {
			err = e
		}
	}()

	cmc = &CommonModelConfig{}
	err = json.NewDecoder(configFile).Decode(&cmc)
	if err != nil {
		return nil, fmt.Errorf("cannot parse JSON config file: %w", err)
	}
	return cmc, nil
}
