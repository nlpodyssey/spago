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
	Activation activation.Name
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T float.DType](in int, activation activation.Name) *Model {
	return &Model{
		WIn:        nn.NewParam(mat.NewEmptyDense[T](in, in)),
		BIn:        nn.NewParam(mat.NewEmptyVecDense[T](in)),
		WT:         nn.NewParam(mat.NewEmptyDense[T](in, in)),
		BT:         nn.NewParam(mat.NewEmptyVecDense[T](in)),
		Activation: activation,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = m.forward(x)
	}
	return ys
}

// t = sigmoid(wT (dot) x + bT)
// h = f(wIn (dot) x + bIn)
// y = t * h + (1 - t) * x
func (m *Model) forward(x ag.Node) ag.Node {
	t := ag.Sigmoid(ag.Affine(m.BT, m.WT, x))
	h := activation.Do(m.Activation, ag.Affine(m.BIn, m.WIn, x))
	y := ag.Add(ag.Prod(t, h), ag.Prod(ag.ReverseSub(t, x.Value().NewScalar(1)), x))
	return y
}
