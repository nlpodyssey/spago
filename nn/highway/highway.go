// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package highway

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model[float32] = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	WIn        nn.Param[T] `spago:"type:weights"`
	BIn        nn.Param[T] `spago:"type:biases"`
	WT         nn.Param[T] `spago:"type:weights"`
	BT         nn.Param[T] `spago:"type:biases"`
	Activation ag.OpName
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](in int, activation ag.OpName) *Model[T] {
	return &Model[T]{
		WIn:        nn.NewParam[T](mat.NewEmptyDense[T](in, in)),
		BIn:        nn.NewParam[T](mat.NewEmptyVecDense[T](in)),
		WT:         nn.NewParam[T](mat.NewEmptyDense[T](in, in)),
		BT:         nn.NewParam[T](mat.NewEmptyVecDense[T](in)),
		Activation: activation,
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

// t = sigmoid(wT (dot) x + bT)
// h = f(wIn (dot) x + bIn)
// y = t * h + (1 - t) * x
func (m *Model[T]) forward(x ag.Node[T]) ag.Node[T] {
	g := m.Graph()
	t := g.Sigmoid(g.Affine(m.BT, m.WT, x))
	h := g.Invoke(m.Activation, g.Affine(m.BIn, m.WIn, x))
	y := g.Add(g.Prod(t, h), g.Prod(g.ReverseSub(t, g.NewScalar(1.0)), x))
	return y
}
