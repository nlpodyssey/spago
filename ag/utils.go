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
func Pad[T mat.DType](xs []Node[T], seqLen int, padding func(i int) Node[T]) []Node[T] {
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
func SeparateMatrix[T mat.DType](x Node[T]) [][]Node[T] {
	rows, cols := x.Value().Dims()
	ys := make([][]Node[T], rows)
	for i := range ys {
		row := make([]Node[T], cols)
		for j := range row {
			row[j] = At(x, i, j)
		}
		ys[i] = row
	}
	return ys
}

// SeparateVec returns a slice of Node(s) containing the elements extracted from the input.
// The size of the vector equals the number of input elements.
// You can think of this method as the inverse of the ag.Concat operator.
func SeparateVec[T mat.DType](x Node[T]) []Node[T] {
	size := x.Value().Size()
	ys := make([]Node[T], size)
	for i := 0; i < size; i++ {
		ys[i] = AtVec(x, i)
	}
	return ys
}

// SplitVec splits the x Node into multiple chunks.
func SplitVec[T mat.DType](x Node[T], chunks int) []Node[T] {
	if x.Value().Size()%chunks != 0 {
		panic("nn: incompatible chunks size")
	}
	l := 0
	size := int(mat.Ceil(T(x.Value().Size()) / T(chunks)))
	ys := make([]Node[T], chunks)
	for i := 0; i < chunks; i++ {
		ys[i] = View(x, l, 0, size, 1)
		l += size
	}
	return ys
}

// Sum returns the value that describes the sum of the sample.
// It panics if the input is empty.
func Sum[T mat.DType](xs ...Node[T]) Node[T] {
	sumVector := xs[0]
	for i := 1; i < len(xs); i++ {
		sumVector = Add(sumVector, xs[i])
	}
	return sumVector
}

// Mean returns the value that describes the average of the sample.
func Mean[T mat.DType](xs []Node[T]) Node[T] {
	sumVector := xs[0]
	for i := 1; i < len(xs); i++ {
		sumVector = Add(sumVector, xs[i])
	}
	return DivScalar(sumVector, xs[0].Graph().Constant(T(len(xs))))
}

// Maximum returns the value that describes the maximum of the sample.
func Maximum[T mat.DType](xs []Node[T]) Node[T] {
	maxVector := xs[0]
	for i := 1; i < len(xs); i++ {
		maxVector = Max(maxVector, xs[i])
	}
	return maxVector
}

// Minimum returns the value that describes the minimum of the sample.
func Minimum[T mat.DType](xs []Node[T]) Node[T] {
	minVector := xs[0]
	for i := 1; i < len(xs); i++ {
		minVector = Min(minVector, xs[i])
	}
	return minVector
}

// Affine performs an affine transformation over an arbitrary (odd) number of nodes held in the input.
// The first node is the “bias”, which is added to the output as-is.
// The remaining nodes of the form "Wx" are multiplied together in pairs, then added.
// The pairs except the first whose "x" is nil are not considered.
// y = b + W1x1 + W2x2 + ... + WnXn
func Affine[T mat.DType](xs ...Node[T]) Node[T] {
	if len(xs)%2 == 0 {
		panic("nn: the number of arguments of the affine transformation should be odd")
	}

	// Optimize bounds checks
	x := xs[2]
	w := xs[1]
	y := Add(xs[0], Mul(w, x)) // b + Wx

	for i := 3; i < len(xs)-1; i += 2 {
		w := xs[i]
		x := xs[i+1]
		if x != nil {
			y = Add(y, Mul(w, x))
		}
	}
	return y
}

// BiLinear performs a bilinear transformation of the type (x_1 W x_2)
func BiLinear[DT mat.DType](w, x1, x2 Node[DT]) Node[DT] {
	return Mul(Mul(T(x1), w), x2)
}

// BiAffine performs a biaffine transformation.
func BiAffine[DT mat.DType](w, u, v, b, x1, x2 Node[DT]) Node[DT] {
	return Add(Add(Add(BiLinear(w, x1, x2), Mul(T(u), x1)), Mul(T(v), x2)), b)
}

// PositiveELU returns a new operator node as a result of ELU(x) + 1.
func PositiveELU[T mat.DType](x Node[T]) Node[T] {
	g := x.Graph()
	return AddScalar(ELU(x, g.Constant(1.0)), g.Constant(1.0))
}

// LogSoftmax returns a new operator node as a result of Log(Softmax(x)).
func LogSoftmax[T mat.DType](x Node[T]) Node[T] {
	return Log(Softmax(x))
}

// RowViews calls RowView for each row of x, returning a new slice
// of row-view Nodes.
func RowViews[T mat.DType](x Node[T]) []Node[T] {
	ys := make([]Node[T], x.Value().Rows())
	for i := range ys {
		ys[i] = RowView(x, i)
	}
	return ys
}

// ColViews calls ColView for each column of x, returning a new slice
// of column-view Nodes.
func ColViews[T mat.DType](x Node[T]) []Node[T] {
	ys := make([]Node[T], x.Value().Columns())
	for i := range ys {
		ys[i] = ColView(x, i)
	}
	return ys
}
