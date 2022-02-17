// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

// BiLinear performs a bilinear transformation of the type (x_1 W x_2)
func BiLinear[T mat.DType](w, x1, x2 ag.Node[T]) ag.Node[T] {
	return ag.Mul(ag.Mul(ag.T(x1), w), x2)
}

// BiAffine performs a biaffine transformation.
func BiAffine[T mat.DType](w, u, v, b, x1, x2 ag.Node[T]) ag.Node[T] {
	return ag.Add(ag.Add(ag.Add(BiLinear(w, x1, x2), ag.Mul(ag.T(u), x1)), ag.Mul(ag.T(v), x2)), b)
}

// Conv1D performs a 1D convolution.
func Conv1D[T mat.DType](w, x ag.Node[T], stride int) ag.Node[T] {
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
		ys[i] = ag.Dot(ag.View(x, 0, i*stride, wr, wc), w)
	}
	return ag.Concat(ys...)
}

// Conv2D performs a 2D convolution.
func Conv2D[T mat.DType](w, x ag.Node[T], xStride, yStride int) ag.Node[T] {
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
			var view = ag.View(x, i*xStride, j*yStride, w.Value().Rows(), w.Value().Columns())
			var dotProduct = ag.Dot(view, w)
			outList = append(outList, dotProduct)
		}
	}

	return ag.Reshape(ag.Concat(outList...), dimx, dimy)
}
