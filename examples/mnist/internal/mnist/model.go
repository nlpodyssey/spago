// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mnist

import (
	"golang.org/x/exp/rand"
	"saientist.dev/spago/pkg/ml/act"
	"saientist.dev/spago/pkg/ml/initializers"
	"saientist.dev/spago/pkg/ml/nn"
	"saientist.dev/spago/pkg/ml/nn/cnn"
	"saientist.dev/spago/pkg/ml/nn/convolution"
	"saientist.dev/spago/pkg/ml/nn/perceptron"
	"saientist.dev/spago/pkg/ml/nn/stack"
)

// NewMLP returns a new multi-layer perceptron initialized to zeros.
func NewMLP(in, hidden, out int, hiddenAct, outputAct act.FuncName) *stack.Model {
	return stack.New(
		perceptron.New(in, hidden, hiddenAct),
		perceptron.New(hidden, out, outputAct),
	)
}

// InitRandom initializes the model using the Xavier (Glorot) method.
func InitMLP(model *stack.Model, source rand.Source) {
	for i, layer := range model.Layers {
		var gain float64
		if i == len(model.Layers)-1 { // last layer
			gain = initializers.Gain(act.SoftMax)
		} else {
			gain = initializers.Gain(layer.(*perceptron.Model).Act)
		}
		layer.ForEachParam(func(param *nn.Param) {
			if param.Type() == nn.Weights {
				initializers.XavierUniform(param.Value(), gain, source)
			}
		})
	}
}

// NewCNN returns a new CNN initialized to zeros.
func NewCNN(kernelSizeX, kernelSizeY, inputChannels, outputChannels, maxPoolingRows, maxPoolingCols, hidden, out int, hiddenAct, outputAct act.FuncName) *cnn.Model {
	return cnn.NewModel(
		convolution.New(kernelSizeX, kernelSizeY, 1, 1, inputChannels, outputChannels, hiddenAct),
		maxPoolingRows, maxPoolingCols,
		perceptron.New(hidden, out, outputAct),
	)
}

// InitCNN initializes the model using the Xavier (Glorot) method.
func InitCNN(model *cnn.Model, source rand.Source) {
	for i := 0; i < len(model.Convolution.K); i++ {
		initializers.XavierUniform(model.Convolution.K[i].Value(), initializers.Gain(model.Convolution.Act), source)
		initializers.XavierUniform(model.Convolution.B[i].Value(), initializers.Gain(model.Convolution.Act), source)
	}
	model.FinalLayer.ForEachParam(func(param *nn.Param) {
		if param.Type() == nn.Weights {
			initializers.XavierUniform(param.Value(), initializers.Gain(act.SoftMax), source)
		}
	})
}
