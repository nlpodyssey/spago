// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flatten

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model[float32]{}

// Model is a parameter-free model used to instantiate a new Processor.
type Model[T mat.DType] struct {
	nn.BaseModel
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model.
// TODO: think about possible configurations
func New[T mat.DType]() *Model[T] {
	return &Model[T]{}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	vectorized := func(x ag.Node[T]) ag.Node[T] {
		return ag.T(ag.Flatten(x))
	}
	return []ag.Node[T]{ag.Concat(ag.Map(vectorized, xs)...)}
}
