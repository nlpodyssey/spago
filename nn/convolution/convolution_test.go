// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package convolution

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestConv1D(t *testing.T) {
	t.Run("float32", testConv1D[float32])
	t.Run("float64", testConv1D[float64])
}

func testConv1D[T mat.DType](t *testing.T) {
	x := ag.NewVariable[T](mat.NewDense(3, 4, []T{
		0.2, 0.1, 0.5, 0.8,
		0.4, -0.3, -0.2, -0.3,
		0.5, -0.6, -0.4, 0.6,
	}), true)

	w := ag.NewVariable[T](mat.NewDense(3, 2, []T{
		0.5, -0.4,
		0.3, 0.3,
		0.4, -0.3,
	}), true)

	out := Conv1D(w, x, 1)

	assert.InDeltaSlice(t, []T{
		0.47, -0.42, -0.56,
	}, out.Value().Data(), 0.005)

	ag.Backward[T](out, mat.NewDense(1, 3, []T{1.0, -0.5, -1.0}))

	assert.InDeltaSlice(t, []T{
		-0.35, -0.95,
		0.75, 0.1,
		1.2, -1.0,
	}, w.Grad().Data(), 0.005)

	assert.InDeltaSlice(t, []T{
		0.5, -0.65, -0.3, 0.4,
		0.3, 0.15, -0.45, -0.3,
		0.4, -0.5, -0.25, 0.3,
	}, x.Grad().Data(), 0.005)
}

func TestConv2D(t *testing.T) {
	t.Run("float32", testConv2D[float32])
	t.Run("float64", testConv2D[float64])
}

func testConv2D[T mat.DType](t *testing.T) {
	x := ag.NewVariable[T](mat.NewDense(4, 4, []T{
		0.2, 0.1, 0.5, 0.8,
		0.4, -0.3, -0.2, -0.3,
		0.5, -0.6, -0.4, 0.6,
		-0.3, 0.9, 0.5, 0.5,
	}), true)

	w := ag.NewVariable[T](mat.NewDense(2, 2, []T{
		0.5, -0.4,
		0.3, 0.3,
	}), true)

	out := Conv2D(w, x, 1, 1)

	assert.InDeltaSlice(t, []T{
		0.09, -0.3, -0.22,
		0.29, -0.37, 0.08,
		0.67, 0.28, -0.14,
	}, out.Value().Data(), 0.005)

	ag.Backward[T](out, mat.NewDense(3, 3, []T{
		1.0, -0.5, -1.0,
		0.5, 0.3, 0.5,
		0.2, 0.5, -0.5,
	}))

	assert.InDeltaSlice(t, []T{
		-0.34, -1.93,
		0.76, 0.16,
	}, w.Grad().Data(), 0.005)

	assert.InDeltaSlice(t, []T{
		0.5, -0.65, -0.3, 0.4,
		0.55, 0.1, -0.32, -0.5,
		0.25, 0.41, -0.21, 0.35,
		0.06, 0.21, 0.0, -0.15,
	}, x.Grad().Data(), 0.005)
}

func TestConv2DStride2(t *testing.T) {
	t.Run("float32", testConv2DStride2[float32])
	t.Run("float64", testConv2DStride2[float64])
}

func testConv2DStride2[T mat.DType](t *testing.T) {
	x := ag.NewVariable[T](mat.NewDense(4, 4, []T{
		0.2, 0.1, 0.5, 0.8,
		0.4, -0.3, -0.2, -0.3,
		0.5, -0.6, -0.4, 0.6,
		-0.3, 0.9, 0.5, 0.5,
	}), true)

	w := ag.NewVariable[T](mat.NewDense(2, 2, []T{
		0.5, -0.4,
		0.3, 0.3,
	}), true)

	out := Conv2D(w, x, 2, 2)

	assert.InDeltaSlice(t, []T{
		0.09, -0.22,
		0.67, -0.14,
	}, out.Value().Data(), 0.005)

	ag.Backward[T](out, mat.NewDense(2, 2, []T{
		1.0, -0.5,
		0.5, 0.3,
	}))

	assert.InDeltaSlice(t, []T{
		0.08, -0.42,
		0.5, 0.45,
	}, w.Grad().Data(), 0.005)

	assert.InDeltaSlice(t, []T{
		0.5, -0.4, -0.25, 0.2,
		0.3, 0.3, -0.15, -0.15,
		0.25, -0.2, 0.15, -0.12,
		0.15, 0.15, 0.09, 0.09,
	}, x.Grad().Data(), 0.005)
}
