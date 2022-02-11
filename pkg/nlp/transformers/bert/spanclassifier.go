// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

var (
	_ nn.Model[float32] = &SpanClassifier[float32]{}
)

// SpanClassifierConfig provides configuration settings for a BERT SpanClassifier.
type SpanClassifierConfig struct {
	InputSize int
}

// SpanClassifier implements span classification for extractive question-answering tasks like SQuAD.
// It uses a linear layers to compute "span start logits" and "span end logits".
type SpanClassifier[T mat.DType] struct {
	*linear.Model[T]
}

func init() {
	gob.Register(&SpanClassifier[float32]{})
	gob.Register(&SpanClassifier[float64]{})
}

// NewSpanClassifier returns a new BERT SpanClassifier model.
func NewSpanClassifier[T mat.DType](config SpanClassifierConfig) *SpanClassifier[T] {
	return &SpanClassifier[T]{
		Model: linear.New[T](config.InputSize, 2),
	}
}

// Classify returns the "span start logits" and "span end logits".
func (p *SpanClassifier[T]) Classify(xs []ag.Node[T]) (startLogits, endLogits []ag.Node[T]) {
	for _, y := range p.Forward(xs...) {
		split := nn.SeparateVec(p.Graph(), y)
		startLogits = append(startLogits, split[0])
		endLogits = append(endLogits, split[1])
	}
	return
}
