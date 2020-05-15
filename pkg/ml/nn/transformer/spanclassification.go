// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package transformer

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

type SpanClassificationConfig struct {
	InputSize int
}

// Span classification for extractive question-answering tasks like SQuAD.
// It uses a linear layers to compute "span start logits" and "span end logits".
type SpanClassification struct {
	*linear.Model
}

func NewSpanClassification(config SpanClassificationConfig) *SpanClassification {
	return &SpanClassification{
		Model: linear.New(config.InputSize, 2),
	}
}

type SpanClassificationProcessor struct {
	*linear.Processor
}

// Classify returns the "span start logits" and "span end logits".
func (p *SpanClassificationProcessor) Classify(xs []ag.Node) (startLogits, endLogits []ag.Node) {
	g := p.Graph()
	for _, y := range p.Forward(xs...) {
		split := nn.SeparateVec(g, y)
		startLogits = append(startLogits, split[0])
		endLogits = append(endLogits, split[1])
	}
	return
}
