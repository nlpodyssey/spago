// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mlpmixer

import (
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/linear"
	"testing"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/stretchr/testify/assert"
)

func TestModel_Forward(t *testing.T) {
	t.Run("float32", testMixerBlockForward[float32])
	t.Run("float32", testMixerBlockForwardWithGeLU[float32])
	t.Run("float64", testMixerBlockForward[float64])
	t.Run("float64", testMixerBlockForwardWithGeLU[float64])
}

func testMixerBlockForward[T mat.DType](t *testing.T) {
	model := newTestModel[T]()
	g := ag.NewGraph[T](ag.WithMode[T](ag.Training))
	proc := nn.Bind(model, g)

	x1 := g.NewVariable(mat.NewVecDense([]T{-0.8, -0.9, -0.9}), true)
	x2 := g.NewVariable(mat.NewVecDense([]T{0.8, -0.3, 0.5}), true)
	x3 := g.NewVariable(mat.NewVecDense([]T{-0.2, 0.7, 0.2}), true)
	x4 := g.NewVariable(mat.NewVecDense([]T{-0.6, 0.1, 0.8}), true)
	x5 := g.NewVariable(mat.NewVecDense([]T{0.5, 0.5, 0.1}), true)

	output := proc.Forward(x1, x2, x3, x4, x5)
	assert.InDeltaSlice(t, []T{0.61250253, -0.61697177, 0.5283925}, output[0].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{1.39401254, -1.00455241, -0.125974}, output[1].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{1.18908357, 1.279643450, 0.4901403}, output[2].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{-0.0916925, 0.87924213, 1.8230465}, output[3].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{0.53910204, 0.24096750, 0.9499668}, output[4].Value().Data(), 1.0e-05)
}

func testMixerBlockForwardWithGeLU[T mat.DType](t *testing.T) {
	model := newTestModelGelu[T]()
	g := ag.NewGraph[T](ag.WithMode[T](ag.Training))
	proc := nn.Bind(model, g)

	x1 := g.NewVariable(mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.5}), true)
	x2 := g.NewVariable(mat.NewVecDense([]T{0.4, 0.5, 0.6, 0.1}), true)
	x3 := g.NewVariable(mat.NewVecDense([]T{-0.4, -0.5, -0.6, -0.3}), true)

	output := proc.Forward(x1, x2, x3)
	assert.InDeltaSlice(t, []T{0.1966, 0.6945, 0.9838, 1.0145}, output[0].Value().Data(), 1.0e-03)
	assert.InDeltaSlice(t, []T{0.1640, 0.6489, 0.5476, -0.3227}, output[1].Value().Data(), 1.0e-03)
	assert.InDeltaSlice(t, []T{-0.7089, -0.6511, -1.3840, -0.5825}, output[2].Value().Data(), 1.0e-03)
}

func newTestModel[T mat.DType]() *MixerBlock[T] {
	model := New[T](Config{
		InputSize:               3,
		HiddenSizeTokenMixer:    4,
		HiddenSizeChannelMixer:  4,
		Channels:                5,
		ActFunctionTokenMixer:   activation.Tanh,
		ActFunctionChannelMixer: activation.Tanh,
	})
	model.TokenMixerFF.Model.Layers[0].(*linear.Model[T]).W.Value().SetData([]T{
		0.5, 0.6, -0.8, -0.3, -0.7,
		0.7, -0.4, 0.1, 0.8, -0.9,
		0.7, -0.7, 0.3, 0.4, 1.0,
		0.3, 0.9, -0.9, 0.0, 0.1,
	})
	model.TokenMixerFF.Model.Layers[0].(*linear.Model[T]).B.Value().SetData([]T{0.4, 0.0, -0.3, 0.8})
	model.TokenMixerFF.Model.Layers[2].(*linear.Model[T]).W.Value().SetData([]T{
		0.7, -0.1, -0.6, 0.0,
		0.3, 0.4, 0.8, -0.9,
		0.7, -0.4, 0.3, -0.7,
		0.3, 0.2, 0.1, -0.3,
		0.1, 0.0, -0.8, 0.5,
	})
	model.TokenMixerFF.Model.Layers[2].(*linear.Model[T]).B.Value().SetData([]T{0.6, 0.3, 0.9, 0.8, -0.3})

	model.ChannelMixerFF.Model.Layers[0].(*linear.Model[T]).W.Value().SetData([]T{
		0.2, 0.0, -0.1,
		0.2, -0.1, 0.0,
		0.6, -0.8, 0.0,
		-0.1, -0.2, -0.1,
	})
	model.ChannelMixerFF.Model.Layers[0].(*linear.Model[T]).B.Value().SetData([]T{-0.4, -0.4, -0.5, -0.8})
	model.ChannelMixerFF.Model.Layers[2].(*linear.Model[T]).W.Value().SetData([]T{
		-0.9, -0.4, -0.7, 0.0,
		0.5, 0.2, 0.7, 0.1,
		-0.4, -0.5, 0.8, -0.1,
	})
	model.ChannelMixerFF.Model.Layers[2].(*linear.Model[T]).B.Value().SetData([]T{-0.5, 0.4, 0.1})

	model.TokenLayerNorm.W.Value().SetData([]T{0.6, 0.3, 0.9})
	model.TokenLayerNorm.B.Value().SetData([]T{0.4, -0.3, 0.1})
	model.ChannelLayerNorm.W.Value().SetData([]T{-0.8, -0.2, 0.0})
	model.ChannelLayerNorm.B.Value().SetData([]T{0.8, 0.9, 0.6})

	return model
}

func newTestModelGelu[T mat.DType]() *MixerBlock[T] {
	model := New[T](Config{
		InputSize:               4,
		HiddenSizeTokenMixer:    4,
		HiddenSizeChannelMixer:  4,
		Channels:                3,
		ActFunctionTokenMixer:   activation.GELU,
		ActFunctionChannelMixer: activation.GELU,
	})
	model.TokenMixerFF.Model.Layers[0].(*linear.Model[T]).W.Value().SetData([]T{
		-0.5501, -0.2185, 0.4135,
		-0.0010, 0.2711, 0.2285,
		-0.0786, 0.2479, -0.4105,
		0.1127, 0.5231, 0.3254,
	})
	model.TokenMixerFF.Model.Layers[0].(*linear.Model[T]).B.Value().SetData([]T{-0.4270, -0.1825, 0.2412, -0.2058})
	model.TokenMixerFF.Model.Layers[2].(*linear.Model[T]).W.Value().SetData([]T{
		0.1136, -0.4490, 0.0887, 0.4140,
		-0.2453, 0.4136, 0.3570, -0.1167,
		-0.1264, 0.0561, -0.4304, -0.2422,
	})
	model.TokenMixerFF.Model.Layers[2].(*linear.Model[T]).B.Value().SetData([]T{0.1743, -0.4632, -0.4156})

	model.ChannelMixerFF.Model.Layers[0].(*linear.Model[T]).W.Value().SetData([]T{
		0.3128, -0.1252, -0.1354, -0.0303,
		-0.4723, 0.0339, 0.3345, 0.3320,
		0.2801, 0.2333, 0.1404, 0.0909,
		0.3981, 0.3470, 0.4891, 0.3329,
	})
	model.ChannelMixerFF.Model.Layers[0].(*linear.Model[T]).B.Value().SetData([]T{-0.0408, -0.4873, 0.2798, 0.4100})
	model.ChannelMixerFF.Model.Layers[2].(*linear.Model[T]).W.Value().SetData([]T{
		-0.2669, -0.4191, 0.3017, 0.1028,
		-0.2485, -0.2905, -0.1644, -0.0897,
		-0.4520, 0.4314, 0.0751, -0.2115,
		-0.4676, 0.3695, 0.1510, -0.2781,
	})
	model.ChannelMixerFF.Model.Layers[2].(*linear.Model[T]).B.Value().SetData([]T{0.0767, 0.4476, 0.1588, 0.1684})

	model.TokenLayerNorm.W.Value().SetData([]T{1.0, 1.0, 1.0, 1.0})
	model.TokenLayerNorm.B.Value().SetData([]T{0.0, 0.0, 0.0, 0.0})
	model.ChannelLayerNorm.W.Value().SetData([]T{1.0, 1.0, 1.0, 1.0})
	model.ChannelLayerNorm.B.Value().SetData([]T{0.0, 0.0, 0.0, 0.0})

	return model
}
