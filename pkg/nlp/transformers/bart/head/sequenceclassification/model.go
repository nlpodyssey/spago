// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sequenceclassification

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/config"
)

var (
	_ nn.Model[float32] = &Model[float32]{}
)

// Model is a model for sentence-level classification tasks
// which embeds a BART pre-trained model.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	BART       *bart.Model[T]
	Classifier *Classifier[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new Model for sentence-level classification.
func New[T mat.DType](config config.Config[T], embeddingsPath string) *Model[T] {
	return &Model[T]{
		BART: bart.New[T](config, embeddingsPath),
		Classifier: NewClassifier(ClassifierConfig[T]{
			InputSize:     config.DModel,
			HiddenSize:    config.DModel,
			OutputSize:    config.NumLabels,
			PoolerDropout: config.ClassifierDropout,
		}),
	}
}

// Close closes the BART model's embeddings DB.
func (m *Model[_]) Close() {
	m.BART.Close()
}

// Classify performs the classification using the last transformed state.
func (m *Model[T]) Classify(inputIds []int) ag.Node[T] {
	transformed := m.BART.Process(inputIds)
	sentenceRepresentation := transformed[len(transformed)-1]
	return nn.ToNode[T](m.Classifier.Forward(sentenceRepresentation))
}
