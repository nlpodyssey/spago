// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"fmt"
	"github.com/nlpodyssey/spago/mat"
)

// Map returns a transformed version of xs with all its components modified according to the mapping function.
// It is useful for applying an operator to a sequence of nodes. Keep in mind that using this function has an overhead
// because of the callback, however insignificant compared to mathematical computations.
func Map[T mat.DType](mapping func(Node[T]) Node[T], xs []Node[T]) []Node[T] {
	ys := make([]Node[T], len(xs))
	for i, x := range xs {
		ys[i] = mapping(x)
	}
	return ys
}

// Map2 takes two arguments and applies a mapping function (that must take two arguments) to the items from the two node-slices in parallel.
// It panics if one slice is shorter than the other.
func Map2[T mat.DType](mapping func(a Node[T], b Node[T]) Node[T], xs1 []Node[T], xs2 []Node[T]) []Node[T] {
	if len(xs1) != len(xs2) {
		panic(fmt.Sprintf("ag: arguments must have the same size (%d != %d)", len(xs1), len(xs2)))
	}
	ys := make([]Node[T], len(xs1))
	for i, x1 := range xs1 {
		ys[i] = mapping(x1, xs2[i])
	}
	return ys
}

// Pad down/up samples the input to the given size.
func (g *Graph[T]) Pad(xs []Node[T], seqLen int, padding func(i int) Node[T]) []Node[T] {
	if len(xs) == seqLen {
		return xs
	}
	if len(xs) > seqLen {
		return xs[:seqLen]
	}
	padded := make([]Node[T], seqLen)
	copy(padded[:len(xs)], xs)
	for i := len(xs); i < len(padded); i++ {
		padded[i] = padding(i)
	}
	return padded
}

// SeparateMatrix returns a matrix of Node(s) represented as a slice of slice containing the elements extracted from the input.
// The dimensions of the resulting matrix are the same of the input.
func (g *Graph[T]) SeparateMatrix(x Node[T]) [][]Node[T] {
	rows, cols := x.Value().Dims()
	ys := make([][]Node[T], rows)
	for i := range ys {
		row := make([]Node[T], cols)
		for j := range row {
			row[j] = g.At(x, i, j)
		}
		ys[i] = row
	}
	return ys
}

// SeparateVec returns a slice of Node(s) containing the elements extracted from the input.
// The size of the vector equals the number of input elements.
// You can think of this method as the inverse of the ag.Concat operator.
func (g *Graph[T]) SeparateVec(x Node[T]) []Node[T] {
	size := x.Value().Size()
	ys := make([]Node[T], size)
	for i := 0; i < size; i++ {
		ys[i] = g.AtVec(x, i)
	}
	return ys
}

// SplitVec splits the x Node into multiple chunks.
func (g *Graph[T]) SplitVec(x Node[T], chunks int) []Node[T] {
	if x.Value().Size()%chunks != 0 {
		panic("nn: incompatible chunks size")
	}
	l := 0
	size := int(mat.Ceil(T(x.Value().Size()) / T(chunks)))
	ys := make([]Node[T], chunks)
	for i := 0; i < chunks; i++ {
		ys[i] = g.View(x, l, 0, size, 1)
		l += size
	}
	return ys
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

// PositiveELU returns a new operator node as a result of ELU(x) + 1.
func (g *Graph[T]) PositiveELU(x Node[T]) Node[T] {
	return g.AddScalar(g.ELU(x, g.Constant(1.0)), g.Constant(1.0))
}

// LogSoftmax returns a new operator node as a result of Log(Softmax(x)).
func (g *Graph[T]) LogSoftmax(x Node[T]) Node[T] {
	return g.Log(g.Softmax(x))
}
