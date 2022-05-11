// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package adanorm implements the Adaptive Normalization (AdaNorm) method.
//
// Reference: "Understanding and Improving Layer Normalization" by Jingjing Xu, Xu Sun,
// Zhiyuan Zhang,Guangxiang Zhao, Junyang Lin (2019).
// (https://papers.nips.cc/paper/8689-understanding-and-improving-layer-normalization.pdf)
package adanorm

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model[float32]{}

// Model contains the scaling factor.
type Model[T mat.DType] struct {
	nn.Module
	Scale T
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model.
func New[T mat.DType](scale T) *Model[T] {
	return &Model[T]{
		Scale: scale,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	eps := ag.Constant[T](1e-10)
	one := ag.Constant[T](1.0)
	k := ag.Constant[T](0.1)
	c := ag.Constant[T](m.Scale)
	meanVectors := m.Mean(xs)
	devVectors := m.StdDev(meanVectors, xs)
	zs := make([]ag.Node[T], len(xs))

	for i, x := range xs {
		y := ag.DivScalar(ag.SubScalar(x, meanVectors[i]), ag.Add[T](devVectors[i], eps))
		fi := ag.ProdScalar(ag.ReverseSub(ag.ProdScalar(y, k), one), c)
		zs[i] = ag.Prod(y, ag.StopGrad[T](fi)) // detach the gradient of fi and only treat it as a changeable constant in implementation
	}
	return zs
}

// Mean computes the mean of the input.
func (m *Model[T]) Mean(xs []ag.Node[T]) []ag.Node[T] {
	ys := make([]ag.Node[T], len(xs))
	for i, x := range xs {
		ys[i] = ag.ReduceMean(x)
	}
	return ys
}

// StdDev computes the standard deviation of the input.
func (m *Model[T]) StdDev(meanVectors []ag.Node[T], xs []ag.Node[T]) []ag.Node[T] {
	devVectors := make([]ag.Node[T], len(xs))
	for i, x := range xs {
		diffVector := ag.Square(ag.SubScalar(x, meanVectors[i]))
		devVectors[i] = ag.Sqrt(ag.ReduceMean(diffVector))
	}
	return devVectors
}
