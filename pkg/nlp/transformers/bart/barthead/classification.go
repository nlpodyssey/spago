// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package barthead

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartencoder"
)

var (
	_ nn.Model     = &Classification{}
	_ nn.Processor = &bartencoder.LayerProcessor{}
)

// ClassificationConfig provides configuration settings for a BART head for sentence-level
// Classification model.
type ClassificationConfig struct {
	InputSize     int
	HiddenSize    int
	OutputSize    int
	PoolerDropout float64
}

// Classification is a model for BART head for sentence-level classification tasks.
type Classification struct {
	Config ClassificationConfig
	*stack.Model
}

// NewClassification returns a new Classification.
func NewClassification(config ClassificationConfig) *Classification {
	return &Classification{
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

// ClassificationProcessor implements a nn.Processor for a BERT sentence-level Classification.
type ClassificationProcessor struct {
	*stack.Processor
}

// NewProc returns a new processor to execute the forward step.
func (m *Classification) NewProc(ctx nn.Context) nn.Processor {
	return &ClassificationProcessor{
		Processor: m.Model.NewProc(ctx).(*stack.Processor),
	}
}
