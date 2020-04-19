// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package transformer

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/encoding/pe"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/multiheadattention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/perceptron"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"io"
	"log"
)

var _ nn.Model = &Layer{}

// Transformer's Layer. Each layer has two sub-layers. The first is a multi-head self-attention mechanism,
// and the second  a.k.a. intermediate layer is position-wise fully connected feed-forward network.
// Each of the two sub-layers uses pre-norm residual connections (Toan Q. Nguyen and Julian Salazar, 2019).
type Layer struct {
	MultiHeadAttention    *multiheadattention.Model
	FFN                   *stack.Model
	ResidualWeight        *nn.Param `type:"weights"`
	positionalEncoder     *pe.PositionalEncoder
	depth                 int
	useDepthEncoding      bool
	usePositionalEncoding bool
}

func NewLayer(
	size int,
	numAttentionHeads,
	intermediateSize int,
	intermediateActivation ag.OpName,
	depth int,
	useDepthEncoding bool,
	usePositionalEncoding bool,
) *Layer {
	return &Layer{
		MultiHeadAttention:    multiheadattention.New(size, numAttentionHeads),
		FFN:                   newFFN(size, intermediateSize, size, intermediateActivation),
		ResidualWeight:        nn.NewParam(mat.NewScalar(0.0)),
		positionalEncoder:     pe.New(size, 5000), // TODO: move from here, it doesn't make sense to init the PE on each layer!
		depth:                 depth,
		useDepthEncoding:      useDepthEncoding,
		usePositionalEncoding: usePositionalEncoding,
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

var _ nn.Processor = &LayerProcessor{}

type LayerProcessor struct {
	opt                []interface{}
	model              *Layer
	mode               nn.ProcessingMode
	g                  *ag.Graph
	MultiHeadAttention *multiheadattention.Processor
	FFN                *stack.Processor
	ResidualWeight     ag.Node
	depthEncoding      ag.Node
}

func (m *Layer) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &LayerProcessor{
		model:              m,
		mode:               nn.Training,
		opt:                opt,
		g:                  g,
		MultiHeadAttention: m.MultiHeadAttention.NewProc(g).(*multiheadattention.Processor),
		FFN:                m.FFN.NewProc(g).(*stack.Processor),
		ResidualWeight:     g.NewWrap(m.ResidualWeight),
		depthEncoding:      g.NewVariable(m.positionalEncoder.EncodingAt(m.depth), false),
	}
	p.init(opt)
	return p
}

func (p *LayerProcessor) init(opt []interface{}) {
	if len(opt) > 0 {
		log.Fatal("transformer: invalid init layer options")
	}
}

func (p *LayerProcessor) Model() nn.Model         { return p.model }
func (p *LayerProcessor) Graph() *ag.Graph        { return p.g }
func (p *LayerProcessor) RequiresFullSeq() bool   { return true }
func (p *LayerProcessor) Mode() nn.ProcessingMode { return p.mode }

func (p *LayerProcessor) SetMode(mode nn.ProcessingMode) {
	p.mode = mode
	p.MultiHeadAttention.SetMode(mode)
	p.FFN.SetMode(mode)
}

func (p *LayerProcessor) Reset() { p.init(p.opt) }

// addPositionalEncoding returns the input enriched with positional encoding.
// The positional encoding is summed to the input vectors.
func (p *LayerProcessor) addPositionalEncoding(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for pos, x := range xs {
		ys[pos] = p.g.Add(x, p.g.NewVariable(p.model.positionalEncoder.EncodingAt(pos), false))
	}
	return ys
}

// addDepthEncoding returns the input enriched with depth encoding.
// The depth encoding is summed to the input vectors.
// The depth (a.k.a step) encoding has been introduced in the "Universal Transformers" (https://arxiv.org/pdf/1807.03819.pdf).
func (p *LayerProcessor) addDepthEncoding(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for pos, x := range xs {
		ys[pos] = p.g.Add(x, p.depthEncoding)
	}
	return ys
}

func (p *LayerProcessor) Forward(xs ...ag.Node) []ag.Node {
	enhancedInput := p.enhanceInputWithCoordinates(xs)
	subLayer1 := p.residualConnection(enhancedInput, p.MultiHeadAttention.Forward(enhancedInput...))
	subLayer2 := p.residualConnection(subLayer1, p.FFN.Forward(subLayer1...))
	return subLayer2
}

// residualConnection follows the strategy describe in "ReZero is All You Need: Fast Convergence at Large Depth" by Bachlechner et al., 2020
func (p *LayerProcessor) residualConnection(a, b []ag.Node) []ag.Node {
	length := len(a)
	c := make([]ag.Node, length)
	if length == 0 {
		return c
	}
	_ = b[length-1] // Avoid bounds checks in loop
	for i := range c {
		c[i] = p.g.Add(a[i], p.g.ProdScalar(b[i], p.ResidualWeight))
	}
	return c
}

// enhanceInputWithCoordinates returns the input nodes optionally enriched with positional and depth encodings.
func (p *LayerProcessor) enhanceInputWithCoordinates(xs []ag.Node) []ag.Node {
	return p.useDepthEncoding(p.usePositionalEncoding(xs))
}

// useDepthEncoding returns the input enriched with depth encoding if required, or the input itself.
func (p *LayerProcessor) useDepthEncoding(xs []ag.Node) []ag.Node {
	if p.model.useDepthEncoding {
		return p.addDepthEncoding(xs)
	} else {
		return xs
	}
}

// usePositionalEncoding returns the input enriched with positional encoding if required, or the input itself.
func (p *LayerProcessor) usePositionalEncoding(xs []ag.Node) []ag.Node {
	if p.model.usePositionalEncoding {
		return p.addPositionalEncoding(xs)
	} else {
		return xs
	}
}
