// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package linear

import (
	"encoding/gob"
	"sync"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	W nn.Param `spago:"type:weights"`
	B nn.Param `spago:"type:biases"`
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T float.DType](in, out int) *Model {
	return &Model{
		W: nn.NewParam(mat.NewEmptyDense[T](out, in)),
		B: nn.NewParam(mat.NewEmptyVecDense[T](out)),
	}
}

// WithBiasGrad allows you to enable or disable gradient propagation on bias (enabled by default).
func (m *Model) WithBiasGrad(value bool) *Model {
	m.B.SetRequiresGrad(value)
	return m
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
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
