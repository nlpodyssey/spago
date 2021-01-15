// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package layernorm implements the Layer Normalization (LayerNorm) i method.
//
// Reference: "Layer normalization" by Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton (2016).
// (https://arxiv.org/pdf/1607.06450.pdf)
package layernorm

import (
	"encoding/gob"
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

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New(size int) *Model {
	return &Model{
		W: nn.NewParam(mat.NewEmptyVecDense(size)),
		B: nn.NewParam(mat.NewEmptyVecDense(size)),
	}
}

// Forward performs the forward step for each input node and returns the result.
// y = (x - E\[x\]) / sqrt(VAR\[x\] + [EPS]) * g + b
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	g := m.Graph()
	eps := g.Constant(1e-12) // avoid underflow errors
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		mean := g.ReduceMean(x)
		dev := g.SubScalar(x, mean)
		stdDev := g.Sqrt(g.Add(g.ReduceMean(g.Square(dev)), eps))
		ys[i] = g.Add(g.Prod(g.DivScalar(dev, stdDev), m.W), m.B)
	}
	return ys
}
