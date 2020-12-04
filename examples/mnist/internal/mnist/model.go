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
	"github.com/nlpodyssey/spago/pkg/ml/nn/convolution"
	"github.com/nlpodyssey/spago/pkg/ml/nn/flatten"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/pooling"
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
func NewCNN(
	kernelSizeX, kernelSizeY int,
	inputChannels, outputChannels int,
	maxPoolingRows, maxPoolingCols int,
	hidden, out int,
	hiddenAct ag.OpName,
	outputAct ag.OpName,
) *stack.Model {
	convConfig := convolution.Config{
		KernelSizeX:    kernelSizeX,
		KernelSizeY:    kernelSizeY,
		XStride:        1,
		YStride:        1,
		InputChannels:  inputChannels,
		OutputChannels: outputChannels,
		Activation:     hiddenAct,
	}
	return stack.New(
		convolution.New(convConfig),
		pooling.NewMax(maxPoolingRows, maxPoolingCols),
		flatten.New(),
		linear.New(hidden, out),
		activation.New(outputAct),
	)
}

// InitCNN initializes the model using the Xavier (Glorot) method.
func InitCNN(model *stack.Model, rndGen *rand.LockedRand) {
	nn.ForEachParam(model, func(param *nn.Param) {
		if param.Type() == nn.Weights {
			initializers.XavierUniform(param.Value(), 1.0, rndGen)
		}
	})
}
