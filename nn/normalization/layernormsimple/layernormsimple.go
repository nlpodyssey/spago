// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package layernormsimple implements a simple version of LayerNorm (LayerNorm-simple).
//
// Reference: "Understanding and Improving Layer Normalization" by Jingjing Xu, Xu Sun, Zhiyuan Zhang, Guangxiang Zhao, Junyang Lin (2019).
// (https://papers.nips.cc/paper/8689-understanding-and-improving-layer-normalization.pdf)
package layernormsimple

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// Model is an empty model used to instantiate a new Processor.
type Model struct {
	nn.Module
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model.
func New() *Model {
	return &Model{}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	eps := ag.Var(xs[0].Value().NewScalar(1e-10))
	for i, x := range xs {
		mean := ag.ReduceMean(x)
		dev := ag.SubScalar(x, mean)
		stdDev := ag.Sqrt(ag.ReduceMean(ag.Square(dev)))
		ys[i] = ag.DivScalar(ag.SubScalar(x, mean), ag.Add(stdDev, eps))
	}
	return ys
}
