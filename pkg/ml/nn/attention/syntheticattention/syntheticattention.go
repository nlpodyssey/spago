// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package syntheticattention provides an implementation of the Synthetic Attention described in:
// "SYNTHESIZER: Rethinking Self-Attention in Transformer Models" by Tay et al., 2020.
// (https://arxiv.org/pdf/2005.00743.pdf)
package syntheticattention

import (
	"encoding/gob"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

var (
	_ nn.Model = &Model{}
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	Config
	FFN       *stack.Model
	Value     *linear.Model
	W         nn.Param     `spago:"type:weights"`
	Attention *ContextProb `spago:"scope:processor"`
}

// ContextProb is a pair of Context encodings and Prob attention scores.
type ContextProb struct {
	// Context encodings.
	Context []ag.Node
	// Prob attention scores.
	Prob []mat.Matrix
}

// Config provides configuration settings for a Synthetic Attention Model.
type Config struct {
	InputSize  int
	HiddenSize int
	ValueSize  int
	MaxLength  int
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New(config Config) *Model {
	return &Model{
		Config: config,
		FFN: stack.New(
			linear.New(config.InputSize, config.HiddenSize),
			activation.New(ag.OpReLU),
		),
		W:     nn.NewParam(mat.NewEmptyDense(config.MaxLength, config.HiddenSize)),
		Value: linear.New(config.InputSize, config.ValueSize),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	g := m.Graph()
	length := len(xs)
	context := make([]ag.Node, length)
	prob := make([]mat.Matrix, length)
	values := g.Stack(m.Value.Forward(xs...)...)
	rectified := g.Stack(m.FFN.Forward(xs...)...)
	attentionWeights := m.extractAttentionWeights(length)
	mul := g.Mul(attentionWeights, g.T(rectified))
	for i := 0; i < length; i++ {
		attProb := g.Softmax(g.ColView(mul, i))
		context[i] = g.Mul(g.T(attProb), values)
		prob[i] = attProb.Value()
	}
	m.Attention = &ContextProb{
		Context: context,
		Prob:    prob,
	}
	return context
}

// extractAttentionWeights returns the attention parameters tailored to the sequence length.
func (m *Model) extractAttentionWeights(length int) ag.Node {
	g := m.Graph()
	attentionWeights := make([]ag.Node, length)
	for i := 0; i < length; i++ {
		attentionWeights[i] = g.T(g.RowView(m.W, i))
	}
	return g.Stack(attentionWeights...)
}
