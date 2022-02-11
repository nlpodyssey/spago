// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sequenceclassification

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

var (
	_ nn.Model[float32] = &Classifier[float32]{}
)

// ClassifierConfig provides configuration settings for a BART head for sentence-level
// Classifier model.
type ClassifierConfig[T mat.DType] struct {
	InputSize     int
	HiddenSize    int
	OutputSize    int
	PoolerDropout T
}

// Classifier is a model for BART head for sentence-level classification tasks.
type Classifier[T mat.DType] struct {
	Config ClassifierConfig[T]
	*stack.Model[T]
}

func init() {
	gob.Register(&Classifier[float32]{})
	gob.Register(&Classifier[float64]{})
}

// NewClassifier returns a new Classifier.
func NewClassifier[T mat.DType](config ClassifierConfig[T]) *Classifier[T] {
	return &Classifier[T]{
		Config: config,
		Model: stack.New[T](
			// dropout.New(pooler_dropout),
			linear.New[T](config.InputSize, config.HiddenSize),
			activation.New[T](ag.OpTanh),
			// dropout.New[T(pooler_dropout),
			linear.New[T](config.HiddenSize, config.OutputSize),
		),
	}
}
