// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package linear

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"log"
	"sync"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

type Model struct {
	W *nn.Param `type:"weights"`
	B *nn.Param `type:"biases"`
}

func New(in, out int) *Model {
	return &Model{
		W: nn.NewParam(mat.NewEmptyDense(out, in)),
		B: nn.NewParam(mat.NewEmptyVecDense(out)),
	}
}

type Concurrency struct {
	Value bool
}

const defaultConcurrency = true

type Processor struct {
	nn.BaseProcessor
	w           ag.Node
	b           ag.Node
	Concurrency bool
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: false,
		},
		w:           g.NewWrap(m.W),
		b:           g.NewWrap(m.B),
		Concurrency: defaultConcurrency,
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

// y = w (dot) x + b
func (p *Processor) forward(x ag.Node) ag.Node {
	return nn.Affine(p.Graph, p.b, p.w, x)
}
