// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package highway

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
)

var _ nn.Model = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.Module
	WIn        nn.Param[T] `spago:"type:weights"`
	BIn        nn.Param[T] `spago:"type:biases"`
	WT         nn.Param[T] `spago:"type:weights"`
	BT         nn.Param[T] `spago:"type:biases"`
	Activation activation.Name
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](in int, activation activation.Name) *Model[T] {
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
	t := ag.Sigmoid(ag.Affine[T](m.BT, m.WT, x))
	h := activation.Do(m.Activation, ag.Affine[T](m.BIn, m.WIn, x))
	y := ag.Add(ag.Prod(t, h), ag.Prod(ag.ReverseSub(t, ag.NewScalar[T](1.0)), x))
	return y
}
