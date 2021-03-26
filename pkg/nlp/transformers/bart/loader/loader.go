// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loader

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/config"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/head/conditionalgeneration"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/head/sequenceclassification"
	"github.com/nlpodyssey/spago/pkg/utils"
	"log"
	"path"
)

// Load loads a Model model from file.
func Load(modelPath string) (nn.Model, error) {
	configFilename := path.Join(modelPath, config.DefaultConfigurationFile)
	embeddingsPath := path.Join(modelPath, config.DefaultEmbeddingsStorage)
	modelFilename := path.Join(modelPath, config.DefaultModelFile)

	fmt.Printf("Start loading pre-trained model from \"%s\"\n", modelPath)
	fmt.Printf("[1/2] Loading configuration... ")
	c, err := config.Load(configFilename)
	if err != nil {
		return nil, err
	}
	fmt.Printf("ok\n")

	var model nn.Model
	if len(c.Architecture) == 0 {
		model = bart.New(c, embeddingsPath) // BART base
	} else {
		switch c.Architecture[0] {
		case "BartForSequenceClassification":
			model = sequenceclassification.New(c, embeddingsPath)
		case "MarianMTModel":
			model = conditionalgeneration.New(c, embeddingsPath)
		default:
			panic(fmt.Errorf("bart: unsupported architecture %s", c.Architecture[0]))
		}
	}

	fmt.Printf("[2/2] Loading model weights... ")
	err = utils.DeserializeFromFile(modelFilename, model)
	if err != nil {
		log.Fatal(fmt.Sprintf("bert: error during model deserialization (%s)", err.Error()))
	}
	fmt.Println("ok")

	return model, nil
}
