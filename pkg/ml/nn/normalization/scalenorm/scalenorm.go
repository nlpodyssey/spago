// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scalenorm

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model[float32] = &Model[float32]{}
)

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Gain nn.Param[T] `spago:"type:weights"`
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](size int) *Model[T] {
	return &Model[T]{
		Gain: nn.NewParam[T](mat.NewEmptyVecDense[T](size)),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	g := m.Graph()
	eps := g.Constant(1e-10)
	ys := make([]ag.Node[T], len(xs))
	for i, x := range xs {
		norm := g.Sqrt(g.ReduceSum(g.Square(x)))
		ys[i] = g.Prod(g.DivScalar(x, g.AddScalar(norm, eps)), m.Gain)
	}
	return ys
}
