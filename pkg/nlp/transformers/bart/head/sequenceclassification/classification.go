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
	_ nn.Model = &Classifier{}
)

// ClassifierConfig provides configuration settings for a BART head for sentence-level
// Classifier model.
type ClassifierConfig struct {
	InputSize     int
	HiddenSize    int
	OutputSize    int
	PoolerDropout mat.Float
}

// Classifier is a model for BART head for sentence-level classification tasks.
type Classifier struct {
	Config ClassifierConfig
	*stack.Model
}

func init() {
	gob.Register(&Classifier{})
}

// NewClassifier returns a new Classifier.
func NewClassifier(config ClassifierConfig) *Classifier {
	return &Classifier{
		Config: config,
		Model: stack.New(
			// dropout.New(pooler_dropout),
			linear.New(config.InputSize, config.HiddenSize),
			activation.New(ag.OpTanh),
			// dropout.New(pooler_dropout),
			linear.New(config.HiddenSize, config.OutputSize),
		),
	}
}
