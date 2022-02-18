// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pooling

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &MaxPooling[float32]{}

// MaxPooling is a parameter-free model used to instantiate a new Processor.
type MaxPooling[T mat.DType] struct {
	nn.BaseModel
	Rows    int
	Columns int
}

func init() {
	gob.Register(&MaxPooling[float32]{})
	gob.Register(&MaxPooling[float64]{})
}

// NewMax returns a new model.
func NewMax[T mat.DType](rows, columns int) *MaxPooling[T] {
	return &MaxPooling[T]{
		Rows:    rows,
		Columns: columns,
	}
}

// Forward performs the forward step for each input node and returns the result.
// The max pooling is applied independently to each input.
func (m *MaxPooling[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	pooled := func(x ag.Node[T]) ag.Node[T] {
		return ag.MaxPooling(x, m.Rows, m.Columns)
	}
	return ag.Map(pooled, xs)
}
