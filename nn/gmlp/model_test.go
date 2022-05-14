// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gmlp

import (
	"testing"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/linear"
	"github.com/nlpodyssey/spago/nn/sgu"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestModel_Forward(t *testing.T) {
	t.Run("float32", testModelForward[float32])
	t.Run("float64", testModelForward[float64])
}

func testModelForward[T mat.DType](t *testing.T) {
	model := New[T](Config{
		Dim:        4,
		Depth:      2,
		SeqLen:     2,
		FFMult:     4,
		Activation: activation.Identity,
	})
	assert.NotNil(t, model)

	require.Len(t, model.Layers, 2)
	mat.SetData[T](model.Layers[0].(*Residual[T]).PreNorm.Norm.W.Value(), []T{0.1, 0.2, 0.3, 0.4})
	mat.SetData[T](model.Layers[0].(*Residual[T]).PreNorm.Norm.B.Value(), []T{0.5, 0.6, 0.7, 0.8})

	mat.SetData[T](model.Layers[0].(*Residual[T]).PreNorm.Block.Layers[0].(*linear.Model[T]).W.Value(), []T{
		0.01, 0.02, 0.03, 0.04,
		0.05, 0.06, 0.07, 0.08,
		0.09, 0.10, 0.11, 0.12,
		0.13, 0.14, 0.15, 0.16,
		0.17, 0.18, 0.29, 0.20,
		0.21, 0.22, 0.23, 0.24,
		0.25, 0.26, 0.27, 0.28,
		0.29, 0.30, 0.31, 0.32,
		0.33, 0.34, 0.35, 0.36,
		0.37, 0.38, 0.39, 0.40,
		0.41, 0.42, 0.43, 0.44,
		0.45, 0.46, 0.47, 0.48,
		0.49, 0.50, 0.51, 0.52,
		0.53, 0.54, 0.55, 0.56,
		0.57, 0.58, 0.59, 0.60,
		0.61, 0.62, 0.63, 0.64,
	})
	mat.SetData[T](model.Layers[0].(*Residual[T]).PreNorm.Block.Layers[0].(*linear.Model[T]).B.Value(), []T{
		0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80,
	})

	mat.SetData[T](model.Layers[0].(*Residual[T]).PreNorm.Block.Layers[2].(*sgu.Model[T]).Norm.W.Value(), []T{
		0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7,
	})
	mat.SetData[T](model.Layers[0].(*Residual[T]).PreNorm.Block.Layers[2].(*sgu.Model[T]).Norm.B.Value(), []T{
		0.02, 0.04, 0.06, 0.08, 0.01, 0.03, 0.05, 0.07,
	})

	mat.SetData[T](model.Layers[0].(*Residual[T]).PreNorm.Block.Layers[2].(*sgu.Model[T]).Proj.W.Value(), []T{
		0.41, 0.42,
		0.43, 0.44,
	})
	mat.SetData[T](model.Layers[0].(*Residual[T]).PreNorm.Block.Layers[2].(*sgu.Model[T]).Proj.B.Value(), []T{
		0.48, 0.49,
	})

	mat.SetData[T](model.Layers[0].(*Residual[T]).PreNorm.Block.Layers[3].(*linear.Model[T]).W.Value(), []T{
		0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88,
		0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99,
		0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 0.88,
		0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 0.88, 0.77,
	})
	mat.SetData[T](model.Layers[0].(*Residual[T]).PreNorm.Block.Layers[3].(*linear.Model[T]).B.Value(), []T{
		0.55, 0.66, 0.77, 0.88,
	})

	// ---

	mat.SetData[T](model.Layers[1].(*Residual[T]).PreNorm.Norm.W.Value(), []T{0.9, 0.8, 0.7, 0.6})
	mat.SetData[T](model.Layers[1].(*Residual[T]).PreNorm.Norm.B.Value(), []T{0.5, 0.4, 0.3, 0.2})

	mat.SetData[T](model.Layers[1].(*Residual[T]).PreNorm.Block.Layers[0].(*linear.Model[T]).W.Value(), []T{
		0.99, 0.98, 0.97, 0.96,
		0.95, 0.94, 0.93, 0.92,
		0.91, 0.90, 0.89, 0.88,
		0.87, 0.86, 0.85, 0.84,
		0.83, 0.82, 0.81, 0.80,
		0.79, 0.78, 0.77, 0.76,
		0.75, 0.74, 0.73, 0.72,
		0.71, 0.70, 0.69, 0.68,
		0.67, 0.66, 0.65, 0.64,
		0.63, 0.62, 0.61, 0.60,
		0.59, 0.58, 0.57, 0.56,
		0.55, 0.54, 0.53, 0.52,
		0.51, 0.50, 0.49, 0.48,
		0.47, 0.46, 0.45, 0.44,
		0.43, 0.42, 0.41, 0.40,
		0.39, 0.38, 0.37, 0.36,
	})
	mat.SetData[T](model.Layers[1].(*Residual[T]).PreNorm.Block.Layers[0].(*linear.Model[T]).B.Value(), []T{
		0.35, 0.34, 0.33, 0.32, 0.31, 0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21, 0.20,
	})

	mat.SetData[T](model.Layers[1].(*Residual[T]).PreNorm.Block.Layers[2].(*sgu.Model[T]).Norm.W.Value(), []T{
		0.9, 0.7, 0.5, 0.3, 0.8, 0.6, 0.4, 0.2,
	})
	mat.SetData[T](model.Layers[1].(*Residual[T]).PreNorm.Block.Layers[2].(*sgu.Model[T]).Norm.B.Value(), []T{
		0.09, 0.07, 0.05, 0.03, 0.08, 0.06, 0.04, 0.02,
	})

	mat.SetData[T](model.Layers[1].(*Residual[T]).PreNorm.Block.Layers[2].(*sgu.Model[T]).Proj.W.Value(), []T{
		0.61, 0.62,
		0.63, 0.64,
	})
	mat.SetData[T](model.Layers[1].(*Residual[T]).PreNorm.Block.Layers[2].(*sgu.Model[T]).Proj.B.Value(), []T{
		0.68, 0.69,
	})

	mat.SetData[T](model.Layers[1].(*Residual[T]).PreNorm.Block.Layers[3].(*linear.Model[T]).W.Value(), []T{
		0.99, 0.88, 0.77, 0.66, 0.55, 0.44, 0.33, 0.22,
		0.88, 0.77, 0.66, 0.55, 0.44, 0.33, 0.22, 0.11,
		0.77, 0.66, 0.55, 0.44, 0.33, 0.22, 0.11, 0.22,
		0.66, 0.55, 0.44, 0.33, 0.22, 0.11, 0.22, 0.33,
	})
	mat.SetData[T](model.Layers[1].(*Residual[T]).PreNorm.Block.Layers[3].(*linear.Model[T]).B.Value(), []T{
		0.55, 0.44, 0.33, 0.22,
	})

	w1 := ag.NewVariable(mat.NewVecDense([]T{0.11, 0.12, 0.13, 0.14}), true)
	w2 := ag.NewVariable(mat.NewVecDense([]T{0.21, 0.22, 0.23, 0.24}), true)

	ys := model.Forward(w1, w2)
	require.Len(t, ys, 2)
	require.InDeltaSlice(t, []T{12.033182, 11.811123, 11.153941, 10.22517}, ys[0].Value().Data(), 0.00005)
	require.InDeltaSlice(t, []T{12.44335, 12.219593, 11.541459, 10.580078}, ys[1].Value().Data(), 0.00005)
}
