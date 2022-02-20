// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

var _ Model[float32] = &BaseModel[float32]{}

func init() {
	gob.Register(&BaseModel[float32]{})
	gob.Register(&BaseModel[float64]{})
}

// Model is implemented by all neural network architectures.
type Model[T mat.DType] interface {
	ag.Differentiable[T]
	mustEmbedBaseModel()
}

// BaseModel must be embedded into all neural models.
type BaseModel[T mat.DType] struct {
	ag.DifferentiableModule[T]
}

func (m BaseModel[T]) mustEmbedBaseModel() {}

// StandardModel consists of a model that implements a Forward variadic function that accepts ag.Node and returns a slice of ag.Node.
// It is called StandardModel since this is the most frequent forward method among all implemented neural models.
type StandardModel[T mat.DType] interface {
	Model[T]

	// Forward executes the forward step for each input and returns the result.
	// Recurrent networks, treats the input nodes as a sequence. Differently, feed-forward
	// networks are stateless so every computation is independent and possibly concurrent.
	Forward(xs ...ag.Node[T]) []ag.Node[T]
}
