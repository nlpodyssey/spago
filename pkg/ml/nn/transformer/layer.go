// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package transformer

import (
	"io"
	"log"
	"saientist.dev/spago/pkg/ml/act"
	"saientist.dev/spago/pkg/ml/ag"
	"saientist.dev/spago/pkg/ml/nn"
	"saientist.dev/spago/pkg/ml/nn/multiheadattention"
	"saientist.dev/spago/pkg/ml/nn/normalization/layernorm"
	"saientist.dev/spago/pkg/ml/nn/perceptron"
	"saientist.dev/spago/pkg/ml/nn/stack"
)

// Transformer's Layer. Each layer has two sub-layers. The first is a multi-head self-attention mechanism,
// and the second  a.k.a. intermediate layer is position-wise fully connected feed-forward network.
// Each of the two sub-layers uses residual connection followed by layer normalization.
type Layer struct {
	MultiHeadAttention *multiheadattention.Model
	LayerNorm1         *layernorm.Model
	FFN                *stack.Model
	LayerNorm2         *layernorm.Model
}

func NewLayer(size, numAttentionHeads int, intermediateSize int, intermediateActivation act.FuncName) *Layer {
	return &Layer{
		MultiHeadAttention: multiheadattention.New(size, numAttentionHeads),
		LayerNorm1:         layernorm.New(size),
		FFN:                newFFN(size, intermediateSize, size, intermediateActivation),
		LayerNorm2:         layernorm.New(size),
	}
}

func newFFN(in, hidden, out int, activation act.FuncName) *stack.Model {
	return stack.New(
		perceptron.New(in, hidden, activation),
		perceptron.New(hidden, out, act.Identity),
	)
}

func (m *Layer) ForEachParam(callback func(param *nn.Param)) {
	nn.ForEachParam(m, callback)
}

func (m *Layer) Serialize(w io.Writer) (int, error) {
	return nn.Serialize(m, w)
}

func (m *Layer) Deserialize(r io.Reader) (int, error) {
	return nn.Deserialize(m, r)
}

type LayerProcessor struct {
	opt                []interface{}
	model              *Layer
	g                  *ag.Graph
	MultiHeadAttention *multiheadattention.Processor
	Norm1              *layernorm.Processor
	MLP                *stack.Processor
	Norm2              *layernorm.Processor
}

func (m *Layer) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &LayerProcessor{
		model:              m,
		opt:                opt,
		g:                  g,
		MultiHeadAttention: m.MultiHeadAttention.NewProc(g).(*multiheadattention.Processor),
		Norm1:              m.LayerNorm1.NewProc(g).(*layernorm.Processor),
		MLP:                m.FFN.NewProc(g).(*stack.Processor),
		Norm2:              m.LayerNorm2.NewProc(g).(*layernorm.Processor),
	}
	p.init(opt)
	return p
}

func (p *LayerProcessor) init(opt []interface{}) {
	if len(opt) > 0 {
		log.Fatal("transformer: invalid init layer options")
	}
}

func (p *LayerProcessor) Model() nn.Model       { return p.model }
func (p *LayerProcessor) Graph() *ag.Graph      { return p.g }
func (p *LayerProcessor) RequiresFullSeq() bool { return true }
func (p *LayerProcessor) Reset()                { p.init(p.opt) }

func (p *LayerProcessor) Forward(xs ...ag.Node) []ag.Node {
	l1 := p.subLayer1(xs)
	l2 := p.subLayer2(l1)
	return l2
}

func (p *LayerProcessor) subLayer1(xs []ag.Node) []ag.Node {
	return addAndNorm(p.g, p.Norm1, xs, p.MultiHeadAttention.Forward(xs...))
}

func (p *LayerProcessor) subLayer2(xs []ag.Node) []ag.Node {
	return addAndNorm(p.g, p.Norm2, xs, p.MLP.Forward(xs...))
}

func addAndNorm(g *ag.Graph, normalizer *layernorm.Processor, a, b []ag.Node) []ag.Node {
	return normalizer.Forward(add(g, a, b)...)
}

func add(g *ag.Graph, a, b []ag.Node) []ag.Node {
	c := make([]ag.Node, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = g.Add(a[i], b[i])
	}
	return c
}
