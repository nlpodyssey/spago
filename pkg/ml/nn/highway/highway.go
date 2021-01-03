// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package highway

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model = &Model{}
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	WIn        nn.Param `spago:"type:weights"`
	BIn        nn.Param `spago:"type:biases"`
	WT         nn.Param `spago:"type:weights"`
	BT         nn.Param `spago:"type:biases"`
	Activation ag.OpName
}

// New returns a new model with parameters initialized to zeros.
func New(in int, activation ag.OpName) *Model {
	return &Model{
		WIn:        nn.NewParam(mat.NewEmptyDense(in, in)),
		BIn:        nn.NewParam(mat.NewEmptyVecDense(in)),
		WT:         nn.NewParam(mat.NewEmptyDense(in, in)),
		BT:         nn.NewParam(mat.NewEmptyVecDense(in)),
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
	g := m.Graph()
	t := g.Sigmoid(nn.Affine(g, m.BT, m.WT, x))
	h := g.Invoke(m.Activation, nn.Affine(g, m.BIn, m.WIn, x))
	y := g.Add(g.Prod(t, h), g.Prod(g.ReverseSub(t, g.NewScalar(1.0)), x))
	return y
}
