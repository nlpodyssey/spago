// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package convolution

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

// Conv1D performs a 1D convolution.
func Conv1D(w, x mat.Tensor, stride int) mat.Tensor {
	var dim int
	wr, wc := w.Value().Shape()[0], w.Value().Shape()[1]
	xr, xc := x.Value().Shape()[0], x.Value().Shape()[1]
	if (xc-wc)%stride != 0 {
		panic("Incompatible stride value for columns")
	}
	if xr != wr {
		panic("Incompatible stride value for rows")
	}
	dim = (xc-wc)/stride + 1
	ys := make([]mat.Tensor, dim)
	for i := 0; i < dim; i++ {
		fromCol := i * stride
		ys[i] = ag.Dot(ag.Slice(x, 0, fromCol, wr, fromCol+wc), w)
	}
	return ag.Concat(ys...)
}

// Conv2D performs a 2D convolution.
func Conv2D(w, x mat.Tensor, xStride, yStride int) mat.Tensor {
	var dimx, dimy int
	if (x.Value().Shape()[0]-w.Value().Shape()[0])%xStride != 0 {
		panic("Incompatible stride value for rows")
	}
	if (x.Value().Shape()[1]-w.Value().Shape()[1])%yStride != 0 {
		panic("Incompatible stride value for columns")
	}
	dimx = (x.Value().Shape()[0]-w.Value().Shape()[0])/xStride + 1
	dimy = (x.Value().Shape()[1]-w.Value().Shape()[1])/yStride + 1

	shape := w.Value().Shape()
	wRows, wCols := shape[0], shape[1]

	var outList []mat.Tensor
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
