// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package convolution2d

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
	XStride        int
	YStride        int
	InputChannels  int
	OutputChannels int
	Mask           []int
	DepthWise      bool // Special case od depthwise convolution, where outputchannels == inputchannels
	Activation     activation.Name
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
		kernels[i] = nn.NewParam(mat.NewEmptyDense[T](config.KernelSizeX, config.KernelSizeY)).WithGrad(requireGrad)
		biases[i] = nn.NewParam(mat.NewEmptyVecDense[T](1)).WithGrad(requireGrad)
	}
	return &Model{
		Config: config,
		K:      kernels,
		B:      biases,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, m.Config.OutputChannels)
	for i := range ys {
		ys[i] = m.forward(xs, i)
	}
	return ys
}

func (m *Model) forward(xs []ag.Node, outputChannel int) ag.Node {
	offset := outputChannel * m.Config.InputChannels
	var out ag.Node
	if m.Config.DepthWise {
		out = convolution.Conv2D(m.K[outputChannel], xs[outputChannel], m.Config.XStride, m.Config.YStride)
		out = ag.AddScalar(out, m.B[outputChannel])
	} else {
		for i := 0; i < len(xs); i++ {
			if m.Config.Mask == nil || m.Config.Mask[i] == 1 {
				out = ag.Add(out, convolution.Conv2D(m.K[i+offset], xs[i], m.Config.XStride, m.Config.YStride))
				out = ag.AddScalar(out, m.B[i+offset])
			}
		}
	}
	return activation.Do(m.Config.Activation, out)
}
