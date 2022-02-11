// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package syntheticattention provides an implementation of the Synthetic Attention described in:
// "SYNTHESIZER: Rethinking Self-Attention in Transformer Models" by Tay et al., 2020.
// (https://arxiv.org/pdf/2005.00743.pdf)
package syntheticattention

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
	_ nn.Model[float32] = &Model[float32]{}
)

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Config
	FFN       *stack.Model[T]
	Value     *linear.Model[T]
	W         nn.Param[T]     `spago:"type:weights"`
	Attention *ContextProb[T] `spago:"scope:processor"`
}

// ContextProb is a pair of Context encodings and Prob attention scores.
type ContextProb[T mat.DType] struct {
	// Context encodings.
	Context []ag.Node[T]
	// Prob attention scores.
	Prob []mat.Matrix[T]
}

// Config provides configuration settings for a Synthetic Attention Model.
type Config struct {
	InputSize  int
	HiddenSize int
	ValueSize  int
	MaxLength  int
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](config Config) *Model[T] {
	return &Model[T]{
		Config: config,
		FFN: stack.New[T](
			linear.New[T](config.InputSize, config.HiddenSize),
			activation.New[T](ag.OpReLU),
		),
		W:     nn.NewParam[T](mat.NewEmptyDense[T](config.MaxLength, config.HiddenSize)),
		Value: linear.New[T](config.InputSize, config.ValueSize),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	g := m.Graph()
	length := len(xs)
	context := make([]ag.Node[T], length)
	prob := make([]mat.Matrix[T], length)
	values := g.Stack(m.Value.Forward(xs...)...)
	rectified := g.Stack(m.FFN.Forward(xs...)...)
	attentionWeights := m.extractAttentionWeights(length)
	mul := g.Mul(attentionWeights, g.T(rectified))
	for i := 0; i < length; i++ {
		attProb := g.Softmax(g.ColView(mul, i))
		context[i] = g.Mul(g.T(attProb), values)
		prob[i] = attProb.Value()
	}
	m.Attention = &ContextProb[T]{
		Context: context,
		Prob:    prob,
	}
	return context
}

// extractAttentionWeights returns the attention parameters tailored to the sequence length.
func (m *Model[T]) extractAttentionWeights(length int) ag.Node[T] {
	g := m.Graph()
	attentionWeights := make([]ag.Node[T], length)
	for i := 0; i < length; i++ {
		attentionWeights[i] = g.T(g.RowView(m.W, i))
	}
	return g.Stack(attentionWeights...)
}
