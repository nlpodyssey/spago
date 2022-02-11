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
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model[float32] = &Model[float32]{}
)

// Model is an empty model used to instantiate a new Processor.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model.
func New[T mat.DType]() *Model[T] {
	return &Model[T]{}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	g := m.Graph()
	ys := make([]ag.Node[T], len(xs))
	eps := g.NewScalar(1e-10)
	for i, x := range xs {
		mean := g.ReduceMean(x)
		dev := g.SubScalar(x, mean)
		stdDev := g.Sqrt(g.ReduceMean(g.Square(dev)))
		ys[i] = g.DivScalar(g.SubScalar(x, mean), g.Add(stdDev, eps))
	}
	return ys
}
