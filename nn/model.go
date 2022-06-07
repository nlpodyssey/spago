// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/ag"
)

// Model is implemented by all neural network architectures.
type Model interface {
	mustEmbedModule()
}

// StandardModel consists of a model that implements a Forward variadic function that accepts ag.Node and returns a slice of ag.Node.
// It is called StandardModel since this is the most frequent forward method among all implemented neural models.
type StandardModel interface {
	Model

	// Forward executes the forward step of the model.
	Forward(...ag.Node) []ag.Node
}

// Forward operates on a slice of StandardModel connecting outputs to inputs sequentially for each module following,
// finally returning its output.
func Forward[M StandardModel](ms []M) func(...ag.Node) []ag.Node {
	return func(xs ...ag.Node) []ag.Node {
		for _, m := range ms {
			xs = m.Forward(xs...)
		}
		return xs
	}
}
