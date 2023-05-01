// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scalenorm

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	Gain *nn.Param
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T float.DType](size int) *Model {
	return &Model{
		Gain: nn.NewParam(mat.NewEmptyVecDense[T](size)),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	eps := xs[0].Value().NewScalar(1e-10)
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		norm := ag.Sqrt(ag.ReduceSum(ag.Square(x)))
		ys[i] = ag.Prod(ag.DivScalar(x, ag.AddScalar(norm, eps)), m.Gain)
	}
	return ys
}
