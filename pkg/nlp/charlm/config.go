// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package charlm

import (
	"encoding/json"
	"os"
)

// Config provides configuration settings for a Character-level Language Model.
// TODO: add dropout
type Config struct {
	VocabularySize    int
	EmbeddingSize     int
	HiddenSize        int
	OutputSize        int    // use the projection layer when the output size is > 0
	SequenceSeparator string // empty string is replaced with DefaultSequenceSeparator
	UnknownToken      string // empty string is replaced with DefaultUnknownToken
}

// LoadConfig loads a Config from file.
func LoadConfig(file string) (Config, error) {
	var config Config
	configFile, err := os.Open(file)
	if err != nil {
		return Config{}, err
	}
	defer configFile.Close()
	err = json.NewDecoder(configFile).Decode(&config)
	if err != nil {
		return Config{}, err
	}
	return config, nil
}
