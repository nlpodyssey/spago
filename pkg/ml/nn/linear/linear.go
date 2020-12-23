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
	W nn.Param `type:"weights"`
	B nn.Param `type:"biases"`
}

// Option allows to configure a new Model with your specific needs.
type Option func(*Model)

// BiasGrad allows you to enable or disable gradient propagation on bias (enabled by default).
func BiasGrad(enable bool) Option {
	return func(m *Model) {
		m.B.SetRequiresGrad(enable)
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

// Processor implements the nn.Processor interface for a linear Model.
type Processor struct {
	nn.BaseProcessor
	// whether to enable the concurrent forward computation
	concurrent bool
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	return &Processor{
		BaseProcessor: nn.NewBaseProcessor(m, ctx, false),
		concurrent:    defaultConcurrency, // TODO: from options
	}
}

// SetConcurrentComputations enables or disables the usage of concurrency
// in the Forward method.
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
	m := p.Model.(*Model)
	return nn.Affine(p.Graph, m.B, m.W, x)
}
