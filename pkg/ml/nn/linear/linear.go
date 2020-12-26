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
	_ nn.Module = &Model{}
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	W nn.Param `type:"weights"`
	B nn.Param `type:"biases"`
	// whether to enable the concurrent forward computation
	ConcurrentComputation bool
}

const defaultConcurrency = true

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
		BaseModel:             nn.BaseModel{FullSeqProcessing: false},
		W:                     nn.NewParam(mat.NewEmptyDense(out, in)),
		B:                     nn.NewParam(mat.NewEmptyVecDense(out)),
		ConcurrentComputation: defaultConcurrency,
	}
	for _, option := range options {
		option(model)
	}
	return model
}

// SetConcurrentComputations enables or disables the usage of concurrency
// in the Forward method.
func (m *Model) SetConcurrentComputations(value bool) {
	m.ConcurrentComputation = value
}

// Forward performs the forward step for each input and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	if m.ConcurrentComputation && len(xs) > 1 {
		return m.fwdConcurrent(xs)
	}
	return m.fwdSerial(xs)
}

func (m *Model) fwdSerial(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = m.forward(x)
	}
	return ys
}

func (m *Model) fwdConcurrent(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	var wg sync.WaitGroup
	wg.Add(len(xs))
	for i := range xs {
		go func(i int) {
			defer wg.Done()
			ys[i] = m.forward(xs[i])
		}(i)
	}
	wg.Wait()
	return ys
}

// y = w (dot) x + b
func (m *Model) forward(x ag.Node) ag.Node {
	return nn.Affine(m.Graph(), m.B, m.W, x)
}
