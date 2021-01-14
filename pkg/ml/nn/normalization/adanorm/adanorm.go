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
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model = &Model{}
)

// Model contains the scaling factor.
type Model struct {
	nn.BaseModel
	Scale  mat.Float
	consts consts `spago:"scope:processor"`
}

type consts struct {
	eps ag.Node
	one ag.Node
	k   ag.Node
	c   ag.Node
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model.
func New(scale mat.Float) *Model {
	return &Model{
		Scale: scale,
	}
}

// InitProcessor initializes constants needed by the Forward().
func (m *Model) InitProcessor() {
	g := m.Graph()
	m.consts = consts{
		eps: g.Constant(1e-10),
		one: g.Constant(1.0),
		k:   g.Constant(0.1),
		c:   g.Constant(m.Scale),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	g := m.Graph()
	meanVectors := m.Mean(xs)
	devVectors := m.StdDev(meanVectors, xs)
	zs := make([]ag.Node, len(xs))

	for i, x := range xs {
		y := g.DivScalar(g.SubScalar(x, meanVectors[i]), g.Add(devVectors[i], m.consts.eps))
		fi := g.ProdScalar(g.ReverseSub(g.ProdScalar(y, m.consts.k), m.consts.one), m.consts.c)
		zs[i] = g.Prod(y, g.NewWrapNoGrad(fi)) // detach the gradient of fi and only treat it as a changeable constant in implementation
	}
	return zs
}

// Mean computes the mean of the input.
func (m *Model) Mean(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = m.Graph().ReduceMean(x)
	}
	return ys
}

// StdDev computes the standard deviation of the input.
func (m *Model) StdDev(meanVectors []ag.Node, xs []ag.Node) []ag.Node {
	g := m.Graph()
	devVectors := make([]ag.Node, len(xs))
	for i, x := range xs {
		diffVector := g.Square(g.SubScalar(x, meanVectors[i]))
		devVectors[i] = g.Sqrt(g.ReduceMean(diffVector))
	}
	return devVectors
}
