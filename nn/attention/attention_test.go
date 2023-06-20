// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package attention

import (
	"math"
	"testing"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
)

func TestScaledDotProductAttention(t *testing.T) {
	t.Run("float32", testScaledDotProductAttention[float32])
	t.Run("float64", testScaledDotProductAttention[float64])
}

func testScaledDotProductAttention[T float.DType](t *testing.T) {
	queries := []mat.Tensor{
		mat.NewDense[T](mat.WithBacking([]T{1.1, 0.0, 2.3}), mat.WithGrad(true)),
		mat.NewDense[T](mat.WithBacking([]T{2.2, -0.5, 0.3}), mat.WithGrad(true)),
		mat.NewDense[T](mat.WithBacking([]T{3.2, 0.5, 0.4}), mat.WithGrad(true)),
	}
	keys := mat.NewDense[T](mat.WithShape(3, 3), mat.WithBacking([]T{
		0.0, 1.2, 1.3,
		4.5, 4.3, 0.2,
		2.7, 3.6, 2.1,
	}), mat.WithGrad(true))
	values := mat.NewDense[T](mat.WithShape(3, 3), mat.WithBacking([]T{
		1.2, 2.3, 3.4,
		2.2, 8.5, 0.0,
		2.3, 6.5, 3.5,
	}), mat.WithGrad(true))

	scaleFactor := mat.Scalar(T(1.0 / math.Sqrt(3)))
	results, _ := ScaledDotProductAttention(queries, keys, values, scaleFactor, false)

	if len(results) != 3 {
		t.Error("The attention doesn't have the expected length")
	}
	assert.InDeltaSlice(t, []T{2.22875441063165, 6.68411289826994, 2.82497984315079}, results[0].Value().Data(), 1.0e-5)
	assert.InDeltaSlice(t, []T{2.20637295180029, 8.15650999969648, 0.539678848469417}, results[1].Value().Data(), 1.0e-5)
	assert.InDeltaSlice(t, []T{2.20423303670527, 8.41210390591632, 0.152898186332002}, results[2].Value().Data(), 1.0e-5)
}

//gocyclo:ignore
func TestScaledDotProductAttention2(t *testing.T) {
	t.Run("float32", testScaledDotProductAttention2[float32])
	t.Run("float64", testScaledDotProductAttention2[float64])
}

func testScaledDotProductAttention2[T float.DType](t *testing.T) {
	queries := []mat.Tensor{
		mat.NewDense[T](mat.WithBacking([]T{0.22, 0.3}), mat.WithGrad(true)),
		mat.NewDense[T](mat.WithBacking([]T{-0.17, 0.24}), mat.WithGrad(true)),
		mat.NewDense[T](mat.WithBacking([]T{-0.15, 0.23}), mat.WithGrad(true)),
	}

	keys := mat.NewDense[T](mat.WithShape(3, 2), mat.WithBacking([]T{
		1.66, 0.12,
		0.88, -0.02,
		-0.3, -0.46,
	}), mat.WithGrad(true))
	values := mat.NewDense[T](mat.WithShape(3, 4), mat.WithBacking([]T{
		0.83, 0.7, -0.25, -0.58,
		0.0, 0.2, 0.57, -2.08,
		-0.07, 0.0, 0.29, 0.5,
	}), mat.WithGrad(true))

	scaleFactor := mat.Scalar(T(1.0 / math.Sqrt(2)))

	// == Forward
	results, weights := ScaledDotProductAttention(queries, keys, values, scaleFactor, false)

	if len(results) != 3 {
		t.Error("The results doesn't have the expected length")
	}
	if len(weights) != 3 {
		t.Error("The weights doesn't have the expected length")
	}
	assert.InDeltaSlice(t, []T{0.312291, 0.347165, 0.170855, -0.813202}, results[0].Value().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{0.232861, 0.284047, 0.21555, -0.694914}, results[1].Value().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{0.236194, 0.28672, 0.21373, -0.700304}, results[2].Value().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{0.398142, 0.342329, 0.259529}, weights[0].Value().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{0.310603, 0.333125, 0.356272}, weights[1].Value().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{0.314262, 0.333682, 0.352055}, weights[2].Value().Data(), 1.0e-6)

	// == Backward
	results[0].AccGrad(mat.NewDense[T](mat.WithBacking([]T{0.7, -0.3, -0.7, -0.5})))
	results[1].AccGrad(mat.NewDense[T](mat.WithBacking([]T{-0.8, -0.5, -0.5, 0.1})))
	results[2].AccGrad(mat.NewDense[T](mat.WithBacking([]T{-0.6, -0.5, 0.2, -0.9})))
	ag.Backward(results...)

	assert.InDeltaSlice(t, []T{0.291064, 0.090078}, queries[0].Grad().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{-0.214319, -0.065291}, queries[1].Grad().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{0.084357, 0.057063}, queries[2].Grad().Data(), 1.0e-6)

	assert.InDeltaSlice(t, []T{
		0.06886, -0.025612,
		-0.039958, 0.089393,
		-0.028902, -0.063781,
	}, keys.Grad().Data(), 1.0e-6)

	assert.InDeltaSlice(t, []T{
		-0.15834, -0.431875, -0.371149, -0.450847,
		-0.22708, -0.436103, -0.339456, -0.438166,
		-0.31458, -0.432022, -0.289395, -0.410987,
	}, values.Grad().Data(), 1.0e-6)
}
