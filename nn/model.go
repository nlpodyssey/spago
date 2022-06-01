// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
)

var _ Model = &Module{}

func init() {
	gob.Register(&Module{})
}

// Model is implemented by all neural network architectures.
type Model interface {
	mustEmbedModule()
}

// Module must be embedded into all neural models.
type Module struct{}

func (m Module) mustEmbedModule() {}

// StandardModel consists of a model that implements a Forward variadic function that accepts ag.Node and returns a slice of ag.Node.
// It is called StandardModel since this is the most frequent forward method among all implemented neural models.
type StandardModel interface {
	Model

	// Forward executes the forward step for each input and returns the result.
	Forward(xs ...ag.Node) []ag.Node
}
