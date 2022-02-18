// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package activation

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model[float32] = &Model[float32]{}

// Model contains the activation operator and serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Activation Name
	Params     []nn.Param[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model.
func New[T mat.DType](activation Name, params ...nn.Param[T]) *Model[T] {
	return &Model[T]{
		Activation: activation,
		Params:     params,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	if m.Activation == Identity {
		return xs
	}
	transformed := func(x ag.Node[T]) ag.Node[T] {
		return Do(m.Activation, append([]ag.Node[T]{x}, ag.ToNodes[T](m.Params)...)...)
	}
	return ag.Map(transformed, xs)
}
