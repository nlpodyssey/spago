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
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model[float32] = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	W nn.Param[T] `spago:"type:weights"`
	B nn.Param[T] `spago:"type:biases"`
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](size int) *Model[T] {
	return &Model[T]{
		W: nn.NewParam[T](mat.NewEmptyVecDense[T](size)),
		B: nn.NewParam[T](mat.NewEmptyVecDense[T](size)),
	}
}

// Forward performs the forward step for each input node and returns the result.
// y = (x - E\[x\]) / sqrt(VAR\[x\] + [EPS]) * g + b
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	eps := xs[0].Graph().Constant(1e-12) // avoid underflow errors
	ys := make([]ag.Node[T], len(xs))
	for i, x := range xs {
		mean := ag.ReduceMean(x)
		dev := ag.SubScalar(x, mean)
		stdDev := ag.Sqrt(ag.Add(ag.ReduceMean(ag.Square(dev)), eps))
		ys[i] = ag.Add[T](ag.Prod[T](ag.DivScalar(dev, stdDev), m.W), m.B)
	}
	return ys
}
