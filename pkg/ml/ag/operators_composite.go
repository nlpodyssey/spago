// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import mat "github.com/nlpodyssey/spago/pkg/mat32"

// PositiveELU returns a new operator node as a result of ELU(x) + 1.
func (g *Graph) PositiveELU(x Node) Node {
	return g.AddScalar(g.ELU(x, g.Constant(1.0)), g.Constant(1.0))
}

// LogSoftmax returns a new operator node as a result of Log(Softmax(x)).
func (g *Graph) LogSoftmax(x Node) Node {
	return g.Log(g.Softmax(x))
}

// Sum returns the value that describes the sum of the sample.
// It panics if the input is empty.
func (g *Graph) Sum(xs ...Node) Node {
	sumVector := xs[0]
	for i := 1; i < len(xs); i++ {
		sumVector = g.Add(sumVector, xs[i])
	}
	return sumVector
}

// Mean returns the value that describes the average of the sample.
func (g *Graph) Mean(xs []Node) Node {
	sumVector := xs[0]
	for i := 1; i < len(xs); i++ {
		sumVector = g.Add(sumVector, xs[i])
	}
	return g.DivScalar(sumVector, g.Constant(mat.Float(len(xs))))
}
