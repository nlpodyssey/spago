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

var _ nn.Model = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.Module
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
func (m *Model[T]) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	if len(xs) == 1 {
		ys[0] = ag.Affine(m.B, m.W, xs[0])
		return ys
	}

	var wg sync.WaitGroup
	wg.Add(len(xs))
	for i, x := range xs {
		go func(i int, x ag.Node) {
			ys[i] = ag.Affine(m.B, m.W, x)
			wg.Done()
		}(i, x)
	}
	wg.Wait()
	return ys
}
