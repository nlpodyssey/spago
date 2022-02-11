// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

var (
	_ nn.Model[float32] = &Classifier[float32]{}
)

// ClassifierConfig provides configuration settings for a BERT Classifier.
type ClassifierConfig struct {
	InputSize int
	Labels    []string
}

// Classifier implements a BERT Classifier.
type Classifier[T mat.DType] struct {
	Config ClassifierConfig
	*linear.Model[T]
}

func init() {
	gob.Register(&Classifier[float32]{})
	gob.Register(&Classifier[float64]{})
}

// NewTokenClassifier returns a new BERT Classifier model.
func NewTokenClassifier[T mat.DType](config ClassifierConfig) *Classifier[T] {
	return &Classifier[T]{
		Config: config,
		Model:  linear.New[T](config.InputSize, len(config.Labels)),
	}
}
