// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

var (
	_ nn.Model     = &Classifier{}
	_ nn.Processor = &ClassifierProcessor{}
)

type ClassifierConfig struct {
	InputSize int
	Labels    []string
}

type Classifier struct {
	Config ClassifierConfig
	*linear.Model
}

// NewTokenClassifier returns a new BERT Classifier model.
func NewTokenClassifier(config ClassifierConfig) *Classifier {
	return &Classifier{
		Config: config,
		Model:  linear.New(config.InputSize, len(config.Labels)),
	}
}

type ClassifierProcessor struct {
	*linear.Processor
}

// NewProc returns a new processor to execute the forward step.
func (m *Classifier) NewProc(ctx nn.Context) nn.Processor {
	return &ClassifierProcessor{
		Processor: m.Model.NewProc(ctx).(*linear.Processor),
	}
}

// Predict returns the logits.
func (p *ClassifierProcessor) Predict(xs []ag.Node) []ag.Node {
	return p.Forward(xs...)
}
