// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package transformer

import (
	"io"
	"log"
	"saientist.dev/spago/pkg/ml/ag"
	"saientist.dev/spago/pkg/ml/encoding/pe"
	"saientist.dev/spago/pkg/ml/nn"
	"saientist.dev/spago/pkg/ml/nn/multiheadattention"
	"saientist.dev/spago/pkg/ml/nn/normalization/scalenorm"
	"saientist.dev/spago/pkg/ml/nn/perceptron"
	"saientist.dev/spago/pkg/ml/nn/stack"
)

var _ nn.Model = &Layer{}

// Transformer's Layer. Each layer has two sub-layers. The first is a multi-head self-attention mechanism,
// and the second  a.k.a. intermediate layer is position-wise fully connected feed-forward network.
// Each of the two sub-layers uses pre-norm residual connections (Toan Q. Nguyen and Julian Salazar, 2019).
type Layer struct {
	MultiHeadAttention *multiheadattention.Model
	Norm1              *scalenorm.Model
	FFN                *stack.Model
	Norm2              *scalenorm.Model
	positionalEncoder  *pe.PositionalEncoder
	depth              int
}

func NewLayer(size, numAttentionHeads int, intermediateSize int, intermediateActivation ag.OpName, depth int) *Layer {
	return &Layer{
		MultiHeadAttention: multiheadattention.New(size, numAttentionHeads),
		Norm1:              scalenorm.New(size),
		FFN:                newFFN(size, intermediateSize, size, intermediateActivation),
		Norm2:              scalenorm.New(size),
		positionalEncoder:  pe.New(size, 5000), // TODO: move from here, it doesn't make sense to init the PE on each layer!
		depth:              depth,
	}
}

func newFFN(in, hidden, out int, activation ag.OpName) *stack.Model {
	return stack.New(
		perceptron.New(in, hidden, activation),
		perceptron.New(hidden, out, ag.Identity),
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
	Norm1              *scalenorm.Processor
	MLP                *stack.Processor
	Norm2              *scalenorm.Processor
}

func (m *Layer) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &LayerProcessor{
		model:              m,
		opt:                opt,
		g:                  g,
		MultiHeadAttention: m.MultiHeadAttention.NewProc(g).(*multiheadattention.Processor),
		Norm1:              m.Norm1.NewProc(g).(*scalenorm.Processor),
		MLP:                m.FFN.NewProc(g).(*stack.Processor),
		Norm2:              m.Norm2.NewProc(g).(*scalenorm.Processor),
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

// addCoordinateEncoding returns the input enriched with positional and depth encoding.
// The positional and depth encoding are summed to the input vectors.
// The depth (a.k.a step) encoding has been introduced in the "Universal Transformers" (https://arxiv.org/pdf/1807.03819.pdf).
// From there also the idea of including positional encoding in every layer of the transformer.
func (p *LayerProcessor) addCoordinateEncoding(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	depthEncoding := p.g.NewVariable(p.model.positionalEncoder.EncodingAt(p.model.depth), false)
	for pos, x := range xs {
		ys[pos] = p.g.Add(x, p.g.NewVariable(p.model.positionalEncoder.EncodingAt(pos), false))
		ys[pos] = p.g.Add(ys[pos], depthEncoding)
	}
	return ys
}

func (p *LayerProcessor) Forward(xs ...ag.Node) []ag.Node {
	ce := p.addCoordinateEncoding(xs)
	l1 := p.subLayer1(ce)
	l2 := p.subLayer2(l1)
	return l2
}

func (p *LayerProcessor) subLayer1(xs []ag.Node) []ag.Node {
	return add(p.g, xs, p.MultiHeadAttention.Forward(p.Norm1.Forward(xs...)...))
}

func (p *LayerProcessor) subLayer2(xs []ag.Node) []ag.Node {
	return add(p.g, xs, p.MLP.Forward(p.Norm2.Forward(xs...)...))
}

func add(g *ag.Graph, a, b []ag.Node) []ag.Node {
	c := make([]ag.Node, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = g.Add(a[i], b[i])
	}
	return c
}
