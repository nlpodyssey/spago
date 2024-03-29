// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package convolution1d

import (
	"encoding/gob"
	"fmt"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/convolution"
)

var _ nn.Model = &Model{}

// Config provides configuration settings for a convolution Model.
type Config struct {
	KernelSizeX    int
	KernelSizeY    int
	YStride        int
	InputChannels  int
	OutputChannels int
	Mask           []int
	DepthWise      bool // Special case od depth-wise convolution, where output channels == input channels
	Activation     activation.Activation
}

// Model contains the serializable parameters for a convolutional neural network model.
type Model struct {
	nn.Module
	Config Config
	K      []*nn.Param
	B      []*nn.Param
}

func init() {
	gob.Register(&Model{})
}

// New returns a new convolution Model, initialized according to the given configuration.
func New[T float.DType](config Config) *Model {
	if config.Mask != nil && config.InputChannels != len(config.Mask) {
		panic(fmt.Sprintf("convolution: wrong mask size; found %d, expected %d", config.InputChannels, len(config.Mask)))
	}
	var paramsSize int
	if config.DepthWise {
		if config.OutputChannels != config.InputChannels {
			panic("convolution: DepthWise convolution input channels must be equals to output channels")
		}
		paramsSize = config.OutputChannels
	} else {
		paramsSize = config.InputChannels * config.OutputChannels
	}

	kernels := make([]*nn.Param, paramsSize)
	biases := make([]*nn.Param, paramsSize)
	for i := 0; i < paramsSize; i++ {
		requireGrad := config.Mask == nil || config.Mask[i%len(config.Mask)] == 1
		kernels[i] = nn.NewParam(mat.NewDense[T](mat.WithShape(config.KernelSizeX, config.KernelSizeY))).WithGrad(requireGrad)
		biases[i] = nn.NewParam(mat.NewDense[T](mat.WithShape(1))).WithGrad(requireGrad)
	}
	return &Model{
		Config: config,
		K:      kernels,
		B:      biases,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...mat.Tensor) []mat.Tensor {
	ys := make([]mat.Tensor, m.Config.OutputChannels)
	for i := range ys {
		ys[i] = m.forward(xs, i)
	}
	return ys
}

func (m *Model) forward(xs []mat.Tensor, outputChannel int) mat.Tensor {
	offset := outputChannel * m.Config.InputChannels
	var out mat.Tensor
	if m.Config.DepthWise {
		out = convolution.Conv1D(m.K[outputChannel], xs[outputChannel], m.Config.YStride)
		out = ag.AddScalar(out, m.B[outputChannel])
	} else {
		for i := 0; i < len(xs); i++ {
			if m.Config.Mask == nil || m.Config.Mask[i] == 1 {
				out = ag.Add(out, convolution.Conv1D(m.K[i+offset], xs[i], m.Config.YStride))
				out = ag.AddScalar(out, m.B[i+offset])
			}
		}
	}

	return activation.New(m.Config.Activation).Forward(out)[0] // TODO: refactor for performance
}
