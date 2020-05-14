// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mnist

import (
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/initializers"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/cnn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/convolution"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

// NewMLP returns a new multi-layer perceptron initialized to zeros.
func NewMLP(in, hidden, out int, hiddenAct, outputAct ag.OpName) *stack.Model {
	return stack.New(
		linear.New(in, hidden),
		activation.New(hiddenAct),
		linear.New(hidden, out),
		activation.New(outputAct),
	)
}

// InitRandom initializes the model using the Xavier (Glorot) method.
func InitMLP(model *stack.Model, rndGen *rand.LockedRand) {
	for i := 0; i < len(model.Layers)-1; i += 2 {
		layer := model.Layers[i]
		nextLayer := model.Layers[i+1]
		gain := 1.0
		if nextLayer, ok := nextLayer.(*activation.Model); ok {
			gain = initializers.Gain(nextLayer.Activation)
		}
		nn.ForEachParam(layer, func(param *nn.Param) {
			if param.Type() == nn.Weights {
				initializers.XavierUniform(param.Value(), gain, rndGen)
			}
		})
	}
}

// NewCNN returns a new CNN initialized to zeros.
// TODO: output activation after the cnn Final Layer
func NewCNN(
	kernelSizeX, kernelSizeY int,
	inputChannels, outputChannels int,
	maxPoolingRows, maxPoolingCols int,
	hidden, out int,
	hiddenAct ag.OpName,
	outputAct ag.OpName,
) *cnn.Model {
	// TODO: pass the outputAct to the CNN
	return cnn.NewModel(
		convolution.New(kernelSizeX, kernelSizeY, 1, 1, inputChannels, outputChannels, hiddenAct),
		maxPoolingRows, maxPoolingCols,
		linear.New(hidden, out),
	)
}

// InitCNN initializes the model using the Xavier (Glorot) method.
func InitCNN(model *cnn.Model, rndGen *rand.LockedRand) {
	for i := 0; i < len(model.Convolution.K); i++ {
		initializers.XavierUniform(model.Convolution.K[i].Value(), initializers.Gain(model.Convolution.Activation), rndGen)
		initializers.XavierUniform(model.Convolution.B[i].Value(), initializers.Gain(model.Convolution.Activation), rndGen)
	}
	nn.ForEachParam(model.FinalLayer, func(param *nn.Param) {
		if param.Type() == nn.Weights {
			initializers.XavierUniform(param.Value(), initializers.Gain(ag.OpSoftmax), rndGen)
		}
	})
}
