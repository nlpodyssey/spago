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

var (
	_ nn.Model[float32] = &Model[float32]{}
)

// Model contains the scaling factor.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Scale  T
	consts consts[T] `spago:"scope:processor"`
}

type consts[T mat.DType] struct {
	eps ag.Node[T]
	one ag.Node[T]
	k   ag.Node[T]
	c   ag.Node[T]
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

// InitProcessor initializes constants needed by the Forward().
func (m *Model[T]) InitProcessor() {
	g := m.Graph()
	m.consts = consts[T]{
		eps: g.Constant(1e-10),
		one: g.Constant(1.0),
		k:   g.Constant(0.1),
		c:   g.Constant(m.Scale),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	g := m.Graph()
	meanVectors := m.Mean(xs)
	devVectors := m.StdDev(meanVectors, xs)
	zs := make([]ag.Node[T], len(xs))

	for i, x := range xs {
		y := g.DivScalar(g.SubScalar(x, meanVectors[i]), g.Add(devVectors[i], m.consts.eps))
		fi := g.ProdScalar(g.ReverseSub(g.ProdScalar(y, m.consts.k), m.consts.one), m.consts.c)
		zs[i] = g.Prod(y, g.NewWrapNoGrad(fi)) // detach the gradient of fi and only treat it as a changeable constant in implementation
	}
	return zs
}

// Mean computes the mean of the input.
func (m *Model[T]) Mean(xs []ag.Node[T]) []ag.Node[T] {
	ys := make([]ag.Node[T], len(xs))
	for i, x := range xs {
		ys[i] = m.Graph().ReduceMean(x)
	}
	return ys
}

// StdDev computes the standard deviation of the input.
func (m *Model[T]) StdDev(meanVectors []ag.Node[T], xs []ag.Node[T]) []ag.Node[T] {
	g := m.Graph()
	devVectors := make([]ag.Node[T], len(xs))
	for i, x := range xs {
		diffVector := g.Square(g.SubScalar(x, meanVectors[i]))
		devVectors[i] = g.Sqrt(g.ReduceMean(diffVector))
	}
	return devVectors
}
