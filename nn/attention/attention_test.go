// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package attention

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestScaledDotProductAttention(t *testing.T) {
	t.Run("float32", testScaledDotProductAttention[float32])
	t.Run("float64", testScaledDotProductAttention[float64])
}

func testScaledDotProductAttention[T mat.DType](t *testing.T) {
	g := ag.NewGraph[T]()

	attIn := QKV[T]{
		Queries: []ag.Node[T]{
			g.NewVariable(mat.NewVecDense([]T{1.1, 0.0, 2.3}), true),
			g.NewVariable(mat.NewVecDense([]T{2.2, -0.5, 0.3}), true),
			g.NewVariable(mat.NewVecDense([]T{3.2, 0.5, 0.4}), true),
		},
		Keys: []ag.Node[T]{
			g.NewVariable(mat.NewVecDense([]T{0.0, 1.2, 1.3}), true),
			g.NewVariable(mat.NewVecDense([]T{4.5, 4.3, 0.2}), true),
			g.NewVariable(mat.NewVecDense([]T{2.7, 3.6, 2.1}), true),
		},
		Values: []ag.Node[T]{
			g.NewVariable(mat.NewVecDense([]T{1.2, 2.3, 3.4}), true),
			g.NewVariable(mat.NewVecDense([]T{2.2, 8.5, 0.0}), true),
			g.NewVariable(mat.NewVecDense([]T{2.3, 6.5, 3.5}), true),
		},
	}

	context, _ := ScaledDotProductAttention(attIn, 1.0/mat.Sqrt[T](3), false)

	if len(context) != 3 {
		t.Error("The attention doesn't have the expected length")
	}
	assert.InDeltaSlice(t, []T{2.22875441063165, 6.68411289826994, 2.82497984315079}, context[0].Value().Data(), 1.0e-5)
	assert.InDeltaSlice(t, []T{2.20637295180029, 8.15650999969648, 0.539678848469417}, context[1].Value().Data(), 1.0e-5)
	assert.InDeltaSlice(t, []T{2.20423303670527, 8.41210390591632, 0.152898186332002}, context[2].Value().Data(), 1.0e-5)
}

//gocyclo:ignore
func TestScaledDotProductAttention2(t *testing.T) {
	t.Run("float32", testScaledDotProductAttention2[float32])
	t.Run("float64", testScaledDotProductAttention2[float64])
}

func testScaledDotProductAttention2[T mat.DType](t *testing.T) {
	g := ag.NewGraph[T]()

	attIn := QKV[T]{
		Queries: []ag.Node[T]{
			g.NewVariable(mat.NewVecDense([]T{0.22, 0.3}), true),
			g.NewVariable(mat.NewVecDense([]T{-0.17, 0.24}), true),
			g.NewVariable(mat.NewVecDense([]T{-0.15, 0.23}), true),
		},
		Keys: []ag.Node[T]{
			g.NewVariable(mat.NewVecDense([]T{1.66, 0.12}), true),
			g.NewVariable(mat.NewVecDense([]T{0.88, -0.02}), true),
			g.NewVariable(mat.NewVecDense([]T{-0.3, -0.46}), true),
		},
		Values: []ag.Node[T]{
			g.NewVariable(mat.NewVecDense([]T{0.83, 0.7, -0.25, -0.58}), true),
			g.NewVariable(mat.NewVecDense([]T{0.0, 0.2, 0.57, -2.08}), true),
			g.NewVariable(mat.NewVecDense([]T{-0.07, 0.0, 0.29, 0.5}), true),
		},
	}

	// == Forward
	context, probs := ScaledDotProductAttention(attIn, 1.0/mat.Sqrt[T](2), false)

	if len(context) != 3 {
		t.Error("The context doesn't have the expected length")
	}
	if len(probs) != 3 {
		t.Error("The probs doesn't have the expected length")
	}
	assert.InDeltaSlice(t, []T{0.312291, 0.347165, 0.170855, -0.813202}, context[0].Value().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{0.232861, 0.284047, 0.21555, -0.694914}, context[1].Value().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{0.236194, 0.28672, 0.21373, -0.700304}, context[2].Value().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{0.398142, 0.342329, 0.259529}, probs[0].Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{0.310603, 0.333125, 0.356272}, probs[1].Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{0.314262, 0.333682, 0.352055}, probs[2].Data(), 1.0e-6)

	// == Backward
	context[0].PropagateGrad(mat.NewVecDense([]T{0.7, -0.3, -0.7, -0.5}))
	context[1].PropagateGrad(mat.NewVecDense([]T{-0.8, -0.5, -0.5, 0.1}))
	context[2].PropagateGrad(mat.NewVecDense([]T{-0.6, -0.5, 0.2, -0.9}))
	g.BackwardAll()

	assert.InDeltaSlice(t, []T{0.291064, 0.090078}, attIn.Queries[0].Grad().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{-0.214319, -0.065291}, attIn.Queries[1].Grad().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{0.084357, 0.057063}, attIn.Queries[2].Grad().Data(), 1.0e-6)

	assert.InDeltaSlice(t, []T{0.06886, -0.025612}, attIn.Keys[0].Grad().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{-0.039958, 0.089393}, attIn.Keys[1].Grad().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{-0.028902, -0.063781}, attIn.Keys[2].Grad().Data(), 1.0e-6)

	assert.InDeltaSlice(t, []T{-0.15834, -0.431875, -0.371149, -0.450847}, attIn.Values[0].Grad().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{-0.22708, -0.436103, -0.339456, -0.438166}, attIn.Values[1].Grad().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{-0.31458, -0.432022, -0.289395, -0.410987}, attIn.Values[2].Grad().Data(), 1.0e-6)
}

func TestLinearAttention(t *testing.T) {
	t.Run("float32", testLinearAttention[float32])
	t.Run("float64", testLinearAttention[float64])
}

func testLinearAttention[T mat.DType](t *testing.T) {
	g := ag.NewGraph[T]()

	attIn := QKV[T]{
		Queries: []ag.Node[T]{
			g.NewVariable(mat.NewVecDense([]T{1.8, 1.35, -1.89}), true),
			g.NewVariable(mat.NewVecDense([]T{0.08, 1.27, -1.06}), true),
			g.NewVariable(mat.NewVecDense([]T{0.28, 0.12, -0.67}), true),
		},
		Keys: []ag.Node[T]{
			g.NewVariable(mat.NewVecDense([]T{0.71, -0.5, -1.58}), true),
			g.NewVariable(mat.NewVecDense([]T{1.43, -0.16, 0.49}), true),
			g.NewVariable(mat.NewVecDense([]T{0.58, -0.27, -0.25}), true),
		},
		Values: []ag.Node[T]{
			g.NewVariable(mat.NewVecDense([]T{0.88, -1.09, -0.45}), true),
			g.NewVariable(mat.NewVecDense([]T{0.43, -0.21, -0.75}), true),
			g.NewVariable(mat.NewVecDense([]T{0.84, 0.01, 0.01}), true),
		},
	}

	defaultMappingFunction := func(x ag.Node[T]) ag.Node[T] {
		return ag.PositiveELU(x)
	}
	output := LinearAttention(attIn, defaultMappingFunction, 1e-12)

	if len(output) != 3 {
		t.Error("The attention doesn't have the expected length")
	}
	assert.InDeltaSlice(t, []T{0.68021652, -0.39977211, -0.44051976}, output[0].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{0.678651, -0.38249578, -0.43479299}, output[1].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{0.6720585, -0.38117003, -0.44469679}, output[2].Value().Data(), 1.0e-05)
}
