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
	_ nn.Model     = &TokenClassifier{}
	_ nn.Processor = &TokenClassifierProcessor{}
)

type TokenClassifierConfig struct {
	InputSize int
	Labels    []string
}

type TokenClassifier struct {
	config TokenClassifierConfig
	*linear.Model
}

func NewTokenClassifier(config TokenClassifierConfig) *TokenClassifier {
	return &TokenClassifier{
		config: config,
		Model:  linear.New(config.InputSize, len(config.Labels)),
	}
}

type TokenClassifierProcessor struct {
	*linear.Processor
}

func (m *TokenClassifier) NewProc(g *ag.Graph) nn.Processor {
	return &TokenClassifierProcessor{
		Processor: m.Model.NewProc(g).(*linear.Processor),
	}
}

// Predicts return the logits.
func (p *TokenClassifierProcessor) Predict(xs []ag.Node) []ag.Node {
	return p.Forward(xs...)
}
