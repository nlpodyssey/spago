// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

// PositiveELU returns a new operator node as a result of ELU(x) + 1.
func (g *Graph) PositiveELU(x Node) Node {
	return g.AddScalar(g.ELU(x, g.Constant(1.0)), g.Constant(1.0))
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
	return g.DivScalar(sumVector, g.Constant(float64(len(xs))))
}

// RotateR performs the right circular shift.
// `i` is the number of places by which the elements are shifted.
// TODO: replace this composite operator with a native auto-grad function
func (g *Graph) RotateR(x Node, i int) Node {
	size := x.Value().Size()
	l := size - i
	a := g.View(x, 0, 0, l, 1)
	b := g.View(x, l, 0, size-l, 1)
	return g.Concat(b, a)
}
