// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package highway

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
)

var _ nn.Model = &Model{}

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	WIn        *nn.Param
	BIn        *nn.Param
	WT         *nn.Param
	BT         *nn.Param
	Activation activation.Activation
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T float.DType](in int, activation activation.Activation) *Model {
	return &Model{
		WIn:        nn.NewParam(mat.NewDense[T](mat.WithShape(in, in))),
		BIn:        nn.NewParam(mat.NewDense[T](mat.WithShape(in))),
		WT:         nn.NewParam(mat.NewDense[T](mat.WithShape(in, in))),
		BT:         nn.NewParam(mat.NewDense[T](mat.WithShape(in))),
		Activation: activation,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.DualValue) []ag.DualValue {
	ys := make([]ag.DualValue, len(xs))
	for i, x := range xs {
		ys[i] = m.forward(x)
	}
	return ys
}

// t = sigmoid(wT (dot) x + bT)
// h = f(wIn (dot) x + bIn)
// y = t * h + (1 - t) * x
func (m *Model) forward(x ag.DualValue) ag.DualValue {
	t := ag.Sigmoid(ag.Affine(m.BT, m.WT, x))
	h := activation.New(m.Activation).Forward(ag.Affine(m.BIn, m.WIn, x))[0] // TODO: refactor for performance
	y := ag.Add(ag.Prod(t, h), ag.Prod(ag.ReverseSub(t, x.Value().NewScalar(1)), x))
	return y
}
