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

var _ nn.Model = &Model{}

// Model contains the scaling factor.
type Model struct {
	nn.Module
	Scale float64
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model.
func New(scale float64) *Model {
	return &Model{
		Scale: scale,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	if len(xs) == 0 {
		return nil
	}
	eps := ag.Var(xs[0].Value().NewScalar(mat.Float(1e-10)))
	one := ag.Var(xs[0].Value().NewScalar(mat.Float(1.0)))
	k := ag.Var(xs[0].Value().NewScalar(mat.Float(0.1)))
	c := ag.Var(xs[0].Value().NewScalar(mat.Float(m.Scale)))
	meanVectors := m.Mean(xs)
	devVectors := m.StdDev(meanVectors, xs)
	zs := make([]ag.Node, len(xs))

	for i, x := range xs {
		y := ag.DivScalar(ag.SubScalar(x, meanVectors[i]), ag.Add(devVectors[i], eps))
		fi := ag.ProdScalar(ag.ReverseSub(ag.ProdScalar(y, k), one), c)
		zs[i] = ag.Prod(y, ag.StopGrad(fi)) // detach the gradient of fi and only treat it as a changeable constant in implementation
	}
	return zs
}

// Mean computes the mean of the input.
func (m *Model) Mean(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = ag.ReduceMean(x)
	}
	return ys
}

// StdDev computes the standard deviation of the input.
func (m *Model) StdDev(meanVectors []ag.Node, xs []ag.Node) []ag.Node {
	devVectors := make([]ag.Node, len(xs))
	for i, x := range xs {
		diffVector := ag.Square(ag.SubScalar(x, meanVectors[i]))
		devVectors[i] = ag.Sqrt(ag.ReduceMean(diffVector))
	}
	return devVectors
}
