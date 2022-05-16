// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mlpmixer

import (
	"testing"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/linear"
	"github.com/stretchr/testify/assert"
)

func TestModel_Forward(t *testing.T) {
	t.Run("float32", testMixerBlockForward[float32])
	t.Run("float32", testMixerBlockForwardWithGeLU[float32])
	t.Run("float64", testMixerBlockForward[float64])
	t.Run("float64", testMixerBlockForwardWithGeLU[float64])
}

func testMixerBlockForward[T float.DType](t *testing.T) {
	model := newTestModel[T]()

	x1 := ag.Var(mat.NewVecDense([]T{-0.8, -0.9, -0.9})).WithGrad(true)
	x2 := ag.Var(mat.NewVecDense([]T{0.8, -0.3, 0.5})).WithGrad(true)
	x3 := ag.Var(mat.NewVecDense([]T{-0.2, 0.7, 0.2})).WithGrad(true)
	x4 := ag.Var(mat.NewVecDense([]T{-0.6, 0.1, 0.8})).WithGrad(true)
	x5 := ag.Var(mat.NewVecDense([]T{0.5, 0.5, 0.1})).WithGrad(true)

	output := model.Forward(x1, x2, x3, x4, x5)
	assert.InDeltaSlice(t, []T{0.61250253, -0.61697177, 0.5283925}, output[0].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{1.39401254, -1.00455241, -0.125974}, output[1].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{1.18908357, 1.279643450, 0.4901403}, output[2].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{-0.0916925, 0.87924213, 1.8230465}, output[3].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{0.53910204, 0.24096750, 0.9499668}, output[4].Value().Data(), 1.0e-05)
}

func testMixerBlockForwardWithGeLU[T float.DType](t *testing.T) {
	model := newTestModelGelu[T]()

	x1 := ag.Var(mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.5})).WithGrad(true)
	x2 := ag.Var(mat.NewVecDense([]T{0.4, 0.5, 0.6, 0.1})).WithGrad(true)
	x3 := ag.Var(mat.NewVecDense([]T{-0.4, -0.5, -0.6, -0.3})).WithGrad(true)

	output := model.Forward(x1, x2, x3)
	assert.InDeltaSlice(t, []T{0.1966, 0.6945, 0.9838, 1.0145}, output[0].Value().Data(), 1.0e-03)
	assert.InDeltaSlice(t, []T{0.1640, 0.6489, 0.5476, -0.3227}, output[1].Value().Data(), 1.0e-03)
	assert.InDeltaSlice(t, []T{-0.7089, -0.6511, -1.3840, -0.5825}, output[2].Value().Data(), 1.0e-03)
}

func newTestModel[T float.DType]() *MixerBlock {
	model := New[T](Config{
		InputSize:               3,
		HiddenSizeTokenMixer:    4,
		HiddenSizeChannelMixer:  4,
		Channels:                5,
		ActFunctionTokenMixer:   activation.Tanh,
		ActFunctionChannelMixer: activation.Tanh,
		Eps:                     1e-12,
	})
	mat.SetData[T](model.TokenMixerFF.Model.Layers[0].(*linear.Model).W.Value(), []T{
		0.5, 0.6, -0.8, -0.3, -0.7,
		0.7, -0.4, 0.1, 0.8, -0.9,
		0.7, -0.7, 0.3, 0.4, 1.0,
		0.3, 0.9, -0.9, 0.0, 0.1,
	})
	mat.SetData[T](model.TokenMixerFF.Model.Layers[0].(*linear.Model).B.Value(), []T{0.4, 0.0, -0.3, 0.8})
	mat.SetData[T](model.TokenMixerFF.Model.Layers[2].(*linear.Model).W.Value(), []T{
		0.7, -0.1, -0.6, 0.0,
		0.3, 0.4, 0.8, -0.9,
		0.7, -0.4, 0.3, -0.7,
		0.3, 0.2, 0.1, -0.3,
		0.1, 0.0, -0.8, 0.5,
	})
	mat.SetData[T](model.TokenMixerFF.Model.Layers[2].(*linear.Model).B.Value(), []T{0.6, 0.3, 0.9, 0.8, -0.3})

	mat.SetData[T](model.ChannelMixerFF.Model.Layers[0].(*linear.Model).W.Value(), []T{
		0.2, 0.0, -0.1,
		0.2, -0.1, 0.0,
		0.6, -0.8, 0.0,
		-0.1, -0.2, -0.1,
	})
	mat.SetData[T](model.ChannelMixerFF.Model.Layers[0].(*linear.Model).B.Value(), []T{-0.4, -0.4, -0.5, -0.8})
	mat.SetData[T](model.ChannelMixerFF.Model.Layers[2].(*linear.Model).W.Value(), []T{
		-0.9, -0.4, -0.7, 0.0,
		0.5, 0.2, 0.7, 0.1,
		-0.4, -0.5, 0.8, -0.1,
	})
	mat.SetData[T](model.ChannelMixerFF.Model.Layers[2].(*linear.Model).B.Value(), []T{-0.5, 0.4, 0.1})

	mat.SetData[T](model.TokenLayerNorm.W.Value(), []T{0.6, 0.3, 0.9})
	mat.SetData[T](model.TokenLayerNorm.B.Value(), []T{0.4, -0.3, 0.1})
	mat.SetData[T](model.ChannelLayerNorm.W.Value(), []T{-0.8, -0.2, 0.0})
	mat.SetData[T](model.ChannelLayerNorm.B.Value(), []T{0.8, 0.9, 0.6})

	return model
}

func newTestModelGelu[T float.DType]() *MixerBlock {
	model := New[T](Config{
		InputSize:               4,
		HiddenSizeTokenMixer:    4,
		HiddenSizeChannelMixer:  4,
		Channels:                3,
		ActFunctionTokenMixer:   activation.GELU,
		ActFunctionChannelMixer: activation.GELU,
		Eps:                     1e-12,
	})
	mat.SetData[T](model.TokenMixerFF.Model.Layers[0].(*linear.Model).W.Value(), []T{
		-0.5501, -0.2185, 0.4135,
		-0.0010, 0.2711, 0.2285,
		-0.0786, 0.2479, -0.4105,
		0.1127, 0.5231, 0.3254,
	})
	mat.SetData[T](model.TokenMixerFF.Model.Layers[0].(*linear.Model).B.Value(), []T{-0.4270, -0.1825, 0.2412, -0.2058})
	mat.SetData[T](model.TokenMixerFF.Model.Layers[2].(*linear.Model).W.Value(), []T{
		0.1136, -0.4490, 0.0887, 0.4140,
		-0.2453, 0.4136, 0.3570, -0.1167,
		-0.1264, 0.0561, -0.4304, -0.2422,
	})
	mat.SetData[T](model.TokenMixerFF.Model.Layers[2].(*linear.Model).B.Value(), []T{0.1743, -0.4632, -0.4156})

	mat.SetData[T](model.ChannelMixerFF.Model.Layers[0].(*linear.Model).W.Value(), []T{
		0.3128, -0.1252, -0.1354, -0.0303,
		-0.4723, 0.0339, 0.3345, 0.3320,
		0.2801, 0.2333, 0.1404, 0.0909,
		0.3981, 0.3470, 0.4891, 0.3329,
	})
	mat.SetData[T](model.ChannelMixerFF.Model.Layers[0].(*linear.Model).B.Value(), []T{-0.0408, -0.4873, 0.2798, 0.4100})
	mat.SetData[T](model.ChannelMixerFF.Model.Layers[2].(*linear.Model).W.Value(), []T{
		-0.2669, -0.4191, 0.3017, 0.1028,
		-0.2485, -0.2905, -0.1644, -0.0897,
		-0.4520, 0.4314, 0.0751, -0.2115,
		-0.4676, 0.3695, 0.1510, -0.2781,
	})
	mat.SetData[T](model.ChannelMixerFF.Model.Layers[2].(*linear.Model).B.Value(), []T{0.0767, 0.4476, 0.1588, 0.1684})

	mat.SetData[T](model.TokenLayerNorm.W.Value(), []T{1.0, 1.0, 1.0, 1.0})
	mat.SetData[T](model.TokenLayerNorm.B.Value(), []T{0.0, 0.0, 0.0, 0.0})
	mat.SetData[T](model.ChannelLayerNorm.W.Value(), []T{1.0, 1.0, 1.0, 1.0})
	mat.SetData[T](model.ChannelLayerNorm.B.Value(), []T{0.0, 0.0, 0.0, 0.0})

	return model
}
