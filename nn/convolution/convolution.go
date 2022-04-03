// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package convolution

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

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
		fromCol := i * stride
		ys[i] = ag.Dot(ag.Slice(x, 0, fromCol, wr, fromCol+wc), w)
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

	wRows, wCols := w.Value().Dims()

	var outList []ag.Node[T]
	for i := 0; i < dimx; i++ {
		for j := 0; j < dimy; j++ {
			fromRow := i * xStride
			fromCol := j * yStride
			var view = ag.Slice(x, fromRow, fromCol, fromRow+wRows, fromCol+wCols)
			var dotProduct = ag.Dot(view, w)
			outList = append(outList, dotProduct)
		}
	}

	return ag.Reshape(ag.Concat(outList...), dimx, dimy)
}
