// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
)

// Affine performs an affine transformation over an arbitrary (odd) number of nodes held in the input.
// The first node is the “bias”, which is added to the output as-is.
// The remaining nodes of the form "Wx" are multiplied together in pairs, then added.
// The pairs except the first whose "x" is nil are not considered.
// y = b + W1x1 + W2x2 + ... + WnXn
func Affine[T mat.DType](g *ag.Graph[T], xs ...ag.Node[T]) ag.Node[T] {
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

// BiLinear performs a bilinear transformation of the type (x_1 W x_2)
func BiLinear[T mat.DType](g *ag.Graph[T], w, x1, x2 ag.Node[T]) ag.Node[T] {
	return g.Mul(g.Mul(g.T(x1), w), x2)
}

// BiAffine performs a biaffine transformation.
func BiAffine[T mat.DType](g *ag.Graph[T], w, u, v, b, x1, x2 ag.Node[T]) ag.Node[T] {
	return g.Add(g.Add(g.Add(BiLinear(g, w, x1, x2), g.Mul(g.T(u), x1)), g.Mul(g.T(v), x2)), b)
}

// Conv1D performs a 1D convolution.
func Conv1D[T mat.DType](g *ag.Graph[T], w, x ag.Node[T], stride int) ag.Node[T] {
	var dim int
	wr, wc := w.Value().Rows(), w.Value().Columns()
	xr, xc := x.Value().Rows(), x.Value().Columns()
	if (xc-wc)%stride != 0 {
		panic("Incompatible stride value for columns")
	}
	if xr != wr {
		panic("Incompatible stride value for rows")
	}
	dim = (xc-wc)/stride + 1
	ys := make([]ag.Node[T], dim)
	for i := 0; i < dim; i++ {
		ys[i] = g.Dot(g.View(x, 0, i*stride, wr, wc), w)
	}
	return g.Concat(ys...)
}

// Conv2D performs a 2D convolution.
func Conv2D[T mat.DType](g *ag.Graph[T], w, x ag.Node[T], xStride, yStride int) ag.Node[T] {
	var dimx, dimy int
	if (x.Value().Rows()-w.Value().Rows())%xStride != 0 {
		panic("Incompatible stride value for rows")
	}
	if (x.Value().Columns()-w.Value().Columns())%yStride != 0 {
		panic("Incompatible stride value for columns")
	}
	dimx = (x.Value().Rows()-w.Value().Rows())/xStride + 1
	dimy = (x.Value().Columns()-w.Value().Columns())/yStride + 1

	var outList []ag.Node[T]
	for i := 0; i < dimx; i++ {
		for j := 0; j < dimy; j++ {
			var view = g.View(x, i*xStride, j*yStride, w.Value().Rows(), w.Value().Columns())
			var dotProduct = g.Dot(view, w)
			outList = append(outList, dotProduct)
		}
	}

	return g.Reshape(g.Concat(outList...), dimx, dimy)
}

// Separate returns a matrix of Node(s) represented as a slice of slice containing the elements extracted from the input.
// The dimensions of the resulting matrix are the same of the input.
func Separate[T mat.DType](g *ag.Graph[T], x ag.Node[T]) [][]ag.Node[T] {
	rows, cols := x.Value().Dims()
	ys := make([][]ag.Node[T], rows)
	for i := range ys {
		row := make([]ag.Node[T], cols)
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
func SeparateVec[T mat.DType](g *ag.Graph[T], x ag.Node[T]) []ag.Node[T] {
	size := x.Value().Size()
	ys := make([]ag.Node[T], size)
	for i := 0; i < size; i++ {
		ys[i] = g.AtVec(x, i)
	}
	return ys
}

// SplitVec splits the x Node into multiple chunks.
func SplitVec[T mat.DType](g *ag.Graph[T], x ag.Node[T], chunks int) []ag.Node[T] {
	if x.Value().Size()%chunks != 0 {
		panic("nn: incompatible chunks size")
	}
	l := 0
	size := int(mat.Ceil(T(x.Value().Size()) / T(chunks)))
	ys := make([]ag.Node[T], chunks)
	for i := 0; i < chunks; i++ {
		ys[i] = g.View(x, l, 0, size, 1)
		l += size
	}
	return ys
}
