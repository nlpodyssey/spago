// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sqrdist

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
	B *nn.Param
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T float.DType](in, rank int) *Model {
	return &Model{
		B: nn.NewParam(mat.NewDense[T](mat.WithShape(rank, in))),
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

func (m *Model) forward(x ag.DualValue) ag.DualValue {
	bh := ag.Mul(m.B, x)
	return ag.Mul(ag.T(bh), bh)
}
