// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scalenorm

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
	Gain nn.Param `spago:"type:weights"`
}

// New returns a new model with parameters initialized to zeros.
func New(size int) *Model {
	return &Model{
		Gain: nn.NewParam(mat.NewEmptyVecDense(size)),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	g := m.Graph()
	eps := g.Constant(1e-10)
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		norm := g.Sqrt(g.ReduceSum(g.Square(x)))
		ys[i] = g.Prod(g.DivScalar(x, g.AddScalar(norm, eps)), m.Gain)
	}
	return ys
}
