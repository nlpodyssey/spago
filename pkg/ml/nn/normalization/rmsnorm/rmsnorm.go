// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rmsnorm implements the Root Mean Square Layer Normalization method.
//
// Reference: "Root Mean Square Layer Normalization" by Biao Zhang and Rico Sennrich (2019).
// (https://arxiv.org/pdf/1910.07467.pdf)
package rmsnorm

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
	W nn.Param `spago:"type:weights"`
	B nn.Param `spago:"type:biases"`
}

// New returns a new model with parameters initialized to zeros.
func New(size int) *Model {
	return &Model{
		W: nn.NewParam(mat.NewEmptyVecDense(size)),
		B: nn.NewParam(mat.NewEmptyVecDense(size)),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	g := m.Graph()
	eps := g.Constant(1e-10)
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		rms := g.Sqrt(g.ReduceMean(g.Square(x)))
		ys[i] = g.Add(g.Prod(g.DivScalar(x, g.AddScalar(rms, eps)), m.W), m.B)
	}
	return ys
}
