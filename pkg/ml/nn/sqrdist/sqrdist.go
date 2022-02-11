// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sqrdist

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
	B nn.Param[T] `spago:"type:weights"`
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](in, rank int) *Model[T] {
	return &Model[T]{
		B: nn.NewParam[T](mat.NewEmptyDense[T](rank, in)),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	ys := make([]ag.Node[T], len(xs))
	for i, x := range xs {
		ys[i] = m.forward(x)
	}
	return ys
}

func (m *Model[T]) forward(x ag.Node[T]) ag.Node[T] {
	g := m.Graph()
	bh := g.Mul(m.B, x)
	return g.Mul(g.T(bh), bh)
}
