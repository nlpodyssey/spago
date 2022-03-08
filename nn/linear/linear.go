// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package linear

import (
	"encoding/gob"
	"sync"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model[float32] = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	W nn.Param[T] `spago:"type:weights"`
	B nn.Param[T] `spago:"type:biases"`
}

// Option allows to configure a new Model with your specific needs.
type Option[T mat.DType] func(*Model[T])

// BiasGrad allows you to enable or disable gradient propagation on bias (enabled by default).
func BiasGrad[T mat.DType](enable bool) Option[T] {
	return func(m *Model[T]) {
		m.B.SetRequiresGrad(enable)
	}
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](in, out int, options ...Option[T]) *Model[T] {
	model := &Model[T]{
		W: nn.NewParam[T](mat.NewEmptyDense[T](out, in)),
		B: nn.NewParam[T](mat.NewEmptyVecDense[T](out)),
	}
	for _, option := range options {
		option(model)
	}
	return model
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	if len(xs) > 1 && m.concurrentComputationEnabled() {
		return m.fwdConcurrent(xs)
	}
	return m.fwdSerial(xs)
}

func (m *Model[T]) concurrentComputationEnabled() bool {
	return m.Session.Graph().ConcurrentComputations() > 1
}

func (m *Model[T]) fwdSerial(xs []ag.Node[T]) []ag.Node[T] {
	ys := make([]ag.Node[T], len(xs))
	for i, x := range xs {
		ys[i] = ag.Affine[T](m.B, m.W, x)
	}
	return ys
}

func (m *Model[T]) fwdConcurrent(xs []ag.Node[T]) []ag.Node[T] {
	ys := make([]ag.Node[T], len(xs))
	var wg sync.WaitGroup
	wg.Add(len(xs))
	for i := range xs {
		go func(i int) {
			defer wg.Done()
			ys[i] = ag.Affine[T](m.B, m.W, xs[i])
		}(i)
	}
	wg.Wait()
	return ys
}
