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

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	W   nn.Param `spago:"type:weights"`
	B   nn.Param `spago:"type:biases"`
	Eps float64
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T float.DType](size int, eps float64) *Model {
	return &Model{
		W:   nn.NewParam(mat.NewEmptyVecDense[T](size)),
		B:   nn.NewParam(mat.NewEmptyVecDense[T](size)),
		Eps: eps,
	}
}

// Forward performs the forward step for each input node and returns the result.
// y = (x - E\[x\]) / sqrt(VAR\[x\] + [EPS]) * g + b
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	if len(xs) == 0 {
		return nil
	}
	eps := ag.Var(xs[0].Value().NewScalar(float.Float(m.Eps)))
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		mean := ag.ReduceMean(x)
		dev := ag.SubScalar(x, mean)
		stdDev := ag.Sqrt(ag.Add(ag.ReduceMean(ag.Square(dev)), eps))
		ys[i] = ag.Add(ag.Prod(ag.DivScalar(dev, stdDev), m.W), m.B)
	}
	return ys
}
