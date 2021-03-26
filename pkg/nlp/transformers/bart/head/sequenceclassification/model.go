// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sequenceclassification

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/config"
)

var (
	_ nn.Model = &Model{}
)

// Model is a model for sentence-level classification tasks
// which embeds a BART pre-trained model.
type Model struct {
	nn.BaseModel
	BART       *bart.Model
	Classifier *Classifier
}

func init() {
	gob.Register(&Model{})
}

// New returns a new Model for sentence-level classification.
func New(config config.Config, embeddingsPath string) *Model {
	return &Model{
		BART: bart.New(config, embeddingsPath),
		Classifier: NewClassifier(ClassifierConfig{
			InputSize:     config.DModel,
			HiddenSize:    config.DModel,
			OutputSize:    config.NumLabels,
			PoolerDropout: config.ClassifierDropout,
		}),
	}
}

// Close closes the BART model's embeddings DB.
func (m *Model) Close() {
	m.BART.Close()
}

// Classify performs the classification using the last transformed state.
func (m *Model) Classify(inputIds []int) ag.Node {
	transformed := m.BART.Process(inputIds)
	sentenceRepresentation := transformed[len(transformed)-1]
	return nn.ToNode(m.Classifier.Forward(sentenceRepresentation))
}
