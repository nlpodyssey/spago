// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package linear

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
	W *nn.Param
	B *nn.Param
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T float.DType](in, out int) *Model {
	return &Model{
		W: nn.NewParam(mat.NewDense[T](mat.WithShape(out, in))),
		B: nn.NewParam(mat.NewDense[T](mat.WithShape(out))),
	}
}

// WithBiasGrad allows you to enable or disable gradient propagation on bias (enabled by default).
func (m *Model) WithBiasGrad(value bool) *Model {
	m.B.SetRequiresGrad(value)
	return m
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...mat.Tensor) []mat.Tensor {
	ys := make([]mat.Tensor, len(xs))
	for i, x := range xs {
		ys[i] = ag.Affine(m.B, m.W, x)
	}
	return ys
}
