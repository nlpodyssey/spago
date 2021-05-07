// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package charlm

import (
	"fmt"
	"log"
	"path"

	"github.com/nlpodyssey/spago/pkg/utils"
)

// LoadModel loads a Model model from file.
func LoadModel(modelPath string) (*Model, error) {
	configFilename := path.Join(modelPath, defaultConfigFilename)
	modelFilename := path.Join(modelPath, defaultModelFilename)

	fmt.Printf("Start loading pre-trained model from \"%s\"\n", modelPath)
	fmt.Printf("[1/2] Loading configuration... ")
	config, err := LoadConfig(configFilename)
	if err != nil {
		return nil, err
	}
	fmt.Printf("ok\n")

	model := New(config)

	fmt.Printf("[2/2] Loading model weights... ")
	err = utils.DeserializeFromFile(modelFilename, model)
	if err != nil {
		log.Fatal(fmt.Sprintf("charlm: error during model deserialization (%s)", err.Error()))
	}
	fmt.Println("ok")

	return model, nil
}
