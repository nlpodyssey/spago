// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

// PositiveELU returns a new operator node as a result of ELU(x) + 1.
func (g *Graph[T]) PositiveELU(x Node[T]) Node[T] {
	return g.AddScalar(g.ELU(x, g.Constant(1.0)), g.Constant(1.0))
}

// LogSoftmax returns a new operator node as a result of Log(Softmax(x)).
func (g *Graph[T]) LogSoftmax(x Node[T]) Node[T] {
	return g.Log(g.Softmax(x))
}

// Sum returns the value that describes the sum of the sample.
// It panics if the input is empty.
func (g *Graph[T]) Sum(xs ...Node[T]) Node[T] {
	sumVector := xs[0]
	for i := 1; i < len(xs); i++ {
		sumVector = g.Add(sumVector, xs[i])
	}
	return sumVector
}

// Mean returns the value that describes the average of the sample.
func (g *Graph[T]) Mean(xs []Node[T]) Node[T] {
	sumVector := xs[0]
	for i := 1; i < len(xs); i++ {
		sumVector = g.Add(sumVector, xs[i])
	}
	return g.DivScalar(sumVector, g.Constant(T(len(xs))))
}

// Affine performs an affine transformation over an arbitrary (odd) number of nodes held in the input.
// The first node is the “bias”, which is added to the output as-is.
// The remaining nodes of the form "Wx" are multiplied together in pairs, then added.
// The pairs except the first whose "x" is nil are not considered.
// y = b + W1x1 + W2x2 + ... + WnXn
func (g *Graph[T]) Affine(xs ...Node[T]) Node[T] {
	if len(xs)%2 == 0 {
		panic("nn: the number of arguments of the affine transformation should be odd")
	}

	// Optimize bounds checks
	x := xs[2]
	w := xs[1]
	y := g.Add(xs[0], g.Mul(w, x)) // b + Wx

	for i := 3; i < len(xs)-1; i += 2 {
		w := xs[i]
		x := xs[i+1]
		if x != nil {
			y = g.Add(y, g.Mul(w, x))
		}
	}
	return y
}
