// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package layernormsimple implements a simple version of LayerNorm (LayerNorm-simple).
//
// Reference: "Understanding and Improving Layer Normalization" by Jingjing Xu, Xu Sun, Zhiyuan Zhang, Guangxiang Zhao, Junyang Lin (2019).
// (https://papers.nips.cc/paper/8689-understanding-and-improving-layer-normalization.pdf)
package layernormsimple

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model = &Model{}
)

// Model is an empty model used to instantiate a new Processor.
type Model struct {
	nn.BaseModel
}

// New returns a new model.
func New() *Model {
	return &Model{}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	g := m.Graph()
	ys := make([]ag.Node, len(xs))
	eps := g.NewScalar(1e-10)
	for i, x := range xs {
		mean := g.ReduceMean(x)
		dev := g.SubScalar(x, mean)
		stdDev := g.Sqrt(g.ReduceMean(g.Square(dev)))
		ys[i] = g.DivScalar(g.SubScalar(x, mean), g.Add(stdDev, eps))
	}
	return ys
}
