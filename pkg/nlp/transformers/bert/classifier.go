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
	config ClassifierConfig
	*linear.Model
}

func NewTokenClassifier(config ClassifierConfig) *Classifier {
	return &Classifier{
		config: config,
		Model:  linear.New(config.InputSize, len(config.Labels)),
	}
}

type ClassifierProcessor struct {
	*linear.Processor
}

func (m *Classifier) NewProc(g *ag.Graph) nn.Processor {
	return &ClassifierProcessor{
		Processor: m.Model.NewProc(g).(*linear.Processor),
	}
}

// Predicts return the logits.
func (p *ClassifierProcessor) Predict(xs []ag.Node) []ag.Node {
	return p.Forward(xs...)
}
