// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package multiheadattention

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"io"
	"log"
	"math"
)

var _ nn.Model = &Model{}

// Multi-Head Attention
type Model struct {
	WQ []*nn.Param `type:"weights"`
	WK []*nn.Param `type:"weights"`
	WV []*nn.Param `type:"weights"`
	WO *nn.Param   `type:"weights"`
	h  int         // number of heads
	dm int         // input/output vectors dimension
	dk int         // hidden vectors dimension (dm/h)
}

// 'dm' is the input/output dimension and 'h' is the number of heads.
func New(dm, h int) *Model {
	WQ := make([]*nn.Param, h)
	WK := make([]*nn.Param, h)
	WV := make([]*nn.Param, h)
	dk := dm / h
	for i := 0; i < h; i++ {
		WQ[i] = nn.NewParam(mat.NewEmptyDense(dk, dm))
		WK[i] = nn.NewParam(mat.NewEmptyDense(dk, dm))
		WV[i] = nn.NewParam(mat.NewEmptyDense(dk, dm))
	}
	return &Model{
		WQ: WQ,
		WK: WK,
		WV: WV,
		WO: nn.NewParam(mat.NewEmptyDense(dm, h*dk)),
		h:  h,
		dm: dm,
		dk: dk,
	}
}

func (m *Model) ForEachParam(callback func(param *nn.Param)) {
	nn.ForEachParam(m, callback)
}

func (m *Model) Serialize(w io.Writer) (int, error) {
	return nn.Serialize(m, w)
}

func (m *Model) Deserialize(r io.Reader) (int, error) {
	return nn.Deserialize(m, r)
}

var _ nn.Processor = &Processor{}

type Processor struct {
	opt   []interface{}
	model *Model
	mode  nn.ProcessingMode
	wQ    []ag.Node
	wK    []ag.Node
	wV    []ag.Node
	wO    ag.Node
	g     *ag.Graph
	Heads []*Head // list of self-attention layers
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	length := m.h
	wQ := make([]ag.Node, length)
	wK := make([]ag.Node, length)
	wV := make([]ag.Node, length)

	if length > 0 {
		// Avoid bounds checks in loop
		mwQ := m.WQ
		mwK := m.WK
		mwV := m.WV
		lastIndex := length - 1
		_ = mwQ[lastIndex]
		_ = mwK[lastIndex]
		_ = mwV[lastIndex]

		for i := 0; i < length; i++ {
			wQ[i] = g.NewWrap(mwQ[i])
			wK[i] = g.NewWrap(mwK[i])
			wV[i] = g.NewWrap(mwV[i])
		}
	}

	p := &Processor{
		model: m,
		mode:  nn.Training,
		opt:   opt,
		wQ:    wQ,
		wK:    wK,
		wV:    wV,
		wO:    g.NewWrap(m.WO),
		g:     g,
	}
	p.init(opt)
	return p
}

func (p *Processor) Model() nn.Model                { return p.model }
func (p *Processor) Graph() *ag.Graph               { return p.g }
func (p *Processor) RequiresFullSeq() bool          { return true }
func (p *Processor) Mode() nn.ProcessingMode        { return p.mode }
func (p *Processor) SetMode(mode nn.ProcessingMode) { p.mode = mode }

func (p *Processor) Reset() {
	p.init(p.opt)
}

func (p *Processor) init(opt []interface{}) {
	if len(opt) > 0 {
		log.Fatal("multiheadattention: invalid init options")
	}
}

type Head struct {
	context []ag.Node
	probs   []mat.Matrix
}

func newHead(context []ag.Node, probs []mat.Matrix) *Head {
	return &Head{
		context: context,
		probs:   probs,
	}
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	p.Heads = p.multiHeadAttention(xs)
	for i := 0; i < len(xs); i++ {
		ys[i] = nn.Linear(p.g, p.wO, p.concatHeadsAt(i))
	}
	return ys
}

func (p *Processor) concatHeadsAt(pos int) ag.Node {
	var buf []ag.Node
	for _, head := range p.Heads {
		buf = append(buf, head.context[pos])
	}
	return p.g.Concat(buf...)
}

func (p *Processor) multiHeadAttention(xs []ag.Node) []*Head {
	heads := make([]*Head, p.model.h)
	for i := range heads {
		heads[i] = newHead(p.selfAttention(xs, i))
	}
	return heads
}

func (p *Processor) selfAttention(xs []ag.Node, hi int) (context []ag.Node, probs []mat.Matrix) {
	qs, ks, vs := p.linearProjection(xs, hi)
	return nn.ScaledDotProductAttention(p.g, qs, ks, vs, math.Sqrt(float64(p.model.dk)))
}

func (p *Processor) linearProjection(xs []ag.Node, hi int) (qs, ks, vs []ag.Node) {
	qs = make([]ag.Node, len(xs))
	ks = make([]ag.Node, len(xs))
	vs = make([]ag.Node, len(xs))
	for i, x := range xs {
		qs[i] = nn.Linear(p.g, p.wQ[hi], x)
		ks[i] = nn.Linear(p.g, p.wK[hi], x)
		vs[i] = nn.Linear(p.g, p.wV[hi], x)
	}
	return
}
