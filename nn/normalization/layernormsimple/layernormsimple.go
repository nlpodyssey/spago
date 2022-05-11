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
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model[float32] = &Model[float32]{}

// Model is an empty model used to instantiate a new Processor.
type Model[T mat.DType] struct {
	nn.Module[T]
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
	ys := make([]ag.Node[T], len(xs))
	eps := ag.Constant[T](1e-10)
	for i, x := range xs {
		mean := ag.ReduceMean(x)
		dev := ag.SubScalar(x, mean)
		stdDev := ag.Sqrt(ag.ReduceMean(ag.Square(dev)))
		ys[i] = ag.DivScalar(ag.SubScalar(x, mean), ag.Add(stdDev, eps))
	}
	return ys
}
