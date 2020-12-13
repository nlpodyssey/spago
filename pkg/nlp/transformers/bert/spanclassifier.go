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
	_ nn.Model     = &SpanClassifier{}
	_ nn.Processor = &SpanClassifierProcessor{}
)

type SpanClassifierConfig struct {
	InputSize int
}

// SpanClassifier implements span classification for extractive question-answering tasks like SQuAD.
// It uses a linear layers to compute "span start logits" and "span end logits".
type SpanClassifier struct {
	*linear.Model
}

func NewSpanClassifier(config SpanClassifierConfig) *SpanClassifier {
	return &SpanClassifier{
		Model: linear.New(config.InputSize, 2),
	}
}

type SpanClassifierProcessor struct {
	*linear.Processor
}

func (m *SpanClassifier) NewProc(ctx nn.Context) nn.Processor {
	return &SpanClassifierProcessor{
		Processor: m.Model.NewProc(ctx).(*linear.Processor),
	}
}

// Classify returns the "span start logits" and "span end logits".
func (p *SpanClassifierProcessor) Classify(xs []ag.Node) (startLogits, endLogits []ag.Node) {
	g := p.GetGraph()
	for _, y := range p.Forward(xs...) {
		split := nn.SeparateVec(g, y)
		startLogits = append(startLogits, split[0])
		endLogits = append(endLogits, split[1])
	}
	return
}
