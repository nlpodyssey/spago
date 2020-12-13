// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package linear

import (
	"sync"

	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the serializable parameters.
type Model struct {
	W *nn.Param `type:"weights"`
	B *nn.Param `type:"biases"`
}

type Option func(*Model)

// BiasGrad allows you to enable or disable gradient propagation on bias (enabled by default).
func BiasGrad(enable bool) Option {
	return func(m *Model) {
		nn.RequiresGrad(enable)(m.B)
	}
}

// New returns a new model with parameters initialized to zeros.
func New(in, out int, options ...Option) *Model {
	model := &Model{
		W: nn.NewParam(mat.NewEmptyDense(out, in)),
		B: nn.NewParam(mat.NewEmptyVecDense(out)),
	}
	for _, option := range options {
		option(model)
	}
	return model
}

const defaultConcurrency = true

type Processor struct {
	nn.BaseProcessor
	w ag.Node
	b ag.Node
	// whether to enable the concurrent forward computation
	concurrent bool
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: false,
		},
		w:          ctx.Graph.NewWrap(m.W),
		b:          ctx.Graph.NewWrap(m.B),
		concurrent: defaultConcurrency, // TODO: from options
	}
}

func (p *Processor) SetConcurrentComputations(value bool) {
	p.concurrent = value
}

// Forward performs the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	if p.concurrent && len(xs) > 1 {
		return p.fwdConcurrent(xs)
	}
	return p.fwdSerial(xs)
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
