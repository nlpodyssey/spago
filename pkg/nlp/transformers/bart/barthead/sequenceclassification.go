// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package barthead

import (
	"encoding/gob"
	"fmt"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartconfig"
	"github.com/nlpodyssey/spago/pkg/utils"
	"log"
	"path"
)

var (
	_ nn.Model = &SequenceClassification{}
)

// SequenceClassification is a model for sentence-level classification tasks
// which embeds a BART pre-trained model.
type SequenceClassification struct {
	nn.BaseModel
	BART           *bart.Model
	Classification *Classification
}

func init() {
	gob.Register(&SequenceClassification{})
}

// NewSequenceClassification returns a new SequenceClassification.
func NewSequenceClassification(config bartconfig.Config, embeddingsPath string) *SequenceClassification {
	return &SequenceClassification{
		BART: bart.New(config, embeddingsPath),
		Classification: NewClassification(ClassificationConfig{
			InputSize:     config.DModel,
			HiddenSize:    config.DModel,
			OutputSize:    config.NumLabels,
			PoolerDropout: config.ClassifierDropout,
		}),
	}
}

// Close closes the BART model's embeddings DB.
func (m *SequenceClassification) Close() {
	m.BART.Close()
}

// LoadModelForSequenceClassification loads a SequenceClassification model from file.
func LoadModelForSequenceClassification(modelPath string) (*SequenceClassification, error) {
	configFilename := path.Join(modelPath, bartconfig.DefaultConfigurationFile)
	embeddingsPath := path.Join(modelPath, bartconfig.DefaultEmbeddingsStorage)
	modelFilename := path.Join(modelPath, bartconfig.DefaultModelFile)

	fmt.Printf("Start loading pre-trained model from \"%s\"\n", modelPath)
	fmt.Printf("[1/2] Loading configuration... ")
	config, err := bartconfig.Load(configFilename)
	if err != nil {
		return nil, err
	}
	fmt.Printf("ok\n")
	model := NewSequenceClassification(config, embeddingsPath)

	fmt.Printf("[2/2] Loading model weights... ")
	err = utils.DeserializeFromFile(modelFilename, model)
	if err != nil {
		log.Fatal(fmt.Sprintf("bert: error during model deserialization (%s)", err.Error()))
	}
	fmt.Println("ok")

	return model, nil
}

// Classify performs the classification using the last transformed state.
func (m *SequenceClassification) Classify(in interface{}) ag.Node {
	inputIds := in.([]int)
	transformed := m.BART.Encode(inputIds)
	sentenceRepresentation := transformed[len(transformed)-1]
	return nn.ToNode(m.Classification.Forward(sentenceRepresentation))
}
