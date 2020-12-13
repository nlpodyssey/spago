// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package syntheticattention provides an implementation of the Synthetic Attention described in:
// "SYNTHESIZER: Rethinking Self-Attention in Transformer Models" by Tay et al., 2020.
// (https://arxiv.org/pdf/2005.00743.pdf)
package syntheticattention

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the serializable parameters.
type Model struct {
	Config
	FFN   *stack.Model
	Value *linear.Model
	W     *nn.Param `type:"weights"`
}

type Config struct {
	InputSize  int
	HiddenSize int
	ValueSize  int
	MaxLength  int
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

type ContextProb struct {
	context []ag.Node
	prob    []mat.Matrix
}

type Processor struct {
	nn.BaseProcessor
	ffn       *stack.Processor
	value     *linear.Processor
	w         ag.Node
	Attention *ContextProb
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		ffn:       m.FFN.NewProc(ctx).(*stack.Processor),
		value:     m.Value.NewProc(ctx).(*linear.Processor),
		w:         ctx.Graph.NewWrap(m.W),
		Attention: nil,
	}
}

// Forward performs the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	g := p.Graph
	length := len(xs)
	context := make([]ag.Node, length)
	prob := make([]mat.Matrix, length)
	values := g.Stack(p.value.Forward(xs...)...)
	rectified := g.Stack(p.ffn.Forward(xs...)...)
	attentionWeights := p.extractAttentionWeights(length)
	mul := g.Mul(attentionWeights, g.T(rectified))
	for i := 0; i < length; i++ {
		attProb := g.Softmax(g.ColView(mul, i))
		context[i] = g.Mul(g.T(attProb), values)
		prob[i] = attProb.Value()
	}
	p.Attention = &ContextProb{
		context: context,
		prob:    prob,
	}
	return context
}

// extractAttentionWeights returns the attention parameters tailored to the sequence length.
func (p *Processor) extractAttentionWeights(length int) ag.Node {
	g := p.Graph
	attentionWeights := make([]ag.Node, length)
	for i := 0; i < length; i++ {
		attentionWeights[i] = g.T(g.RowView(p.w, i))
	}
	return g.Stack(attentionWeights...)
}
