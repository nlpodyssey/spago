// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package perceptron

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"io"
	"log"
	"sync"
)

var _ nn.Model = &Model{}

type Model struct {
	W          *nn.Param `type:"weights"`
	B          *nn.Param `type:"biases"`
	Activation ag.OpName // output activation
}

func New(in, out int, actFunc ag.OpName) *Model {
	return &Model{
		W:          nn.NewParam(mat.NewEmptyDense(out, in)),
		B:          nn.NewParam(mat.NewEmptyVecDense(out)),
		Activation: actFunc,
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

// SetActivation sets the new activation and returns the previous one.
func (m *Model) SetActivation(a ag.OpName) ag.OpName {
	prev := m.Activation
	m.Activation = a
	return prev
}

type Concurrency struct {
	Value bool
}

type Processor struct {
	opt         []interface{}
	model       *Model
	g           *ag.Graph
	w           ag.Node
	b           ag.Node
	Concurrency bool
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		model:       m,
		opt:         opt,
		g:           g,
		w:           g.NewWrap(m.W),
		b:           g.NewWrap(m.B),
		Concurrency: true,
	}
	p.init(opt)
	return p
}

func (p *Processor) init(opt []interface{}) {
	for _, t := range opt {
		switch t := t.(type) {
		case Concurrency:
			p.Concurrency = t.Value
		default:
			log.Fatal("perceptron: invalid init options")
		}
	}
}

func (p *Processor) Model() nn.Model {
	return p.model
}

func (p *Processor) Graph() *ag.Graph {
	return p.g
}

func (p *Processor) RequiresFullSeq() bool {
	return false
}

func (p *Processor) Reset() {
	p.init(p.opt)
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	if p.Concurrency && len(xs) > 1 {
		return p.fwdConcurrent(xs)
	} else {
		return p.fwdSerial(xs)
	}
}

func (p *Processor) fwdSerial(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = p.forward(x)
	}
	return ys
}

func (p *Processor) fwdConcurrent(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	var wg sync.WaitGroup
	wg.Add(len(xs))
	for i := range xs {
		go func(i int) {
			defer wg.Done()
			ys[i] = p.forward(xs[i])
		}(i)
	}
	wg.Wait()
	return ys
}

// y = f(w (dot) x + b)
func (p *Processor) forward(x ag.Node) ag.Node {
	return p.g.Invoke(p.model.Activation, nn.Affine(p.g, p.b, p.w, x))
}
