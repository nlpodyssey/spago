// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package convolution1d

import (
	"encoding/gob"
	"fmt"
	"sync"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model[float32] = &Model[float32]{}

// Config provides configuration settings for a convolution Model.
type Config struct {
	KernelSizeX    int
	KernelSizeY    int
	YStride        int
	InputChannels  int
	OutputChannels int
	Mask           []int
	DepthWise      bool // Special case od depth-wise convolution, where output channels == input channels
	Activation     ag.OpName
}

// Model contains the serializable parameters for a convolutional neural network model.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Config Config
	K      []nn.Param[T] `spago:"type:weights"`
	B      []nn.Param[T] `spago:"type:biases"`
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new convolution Model, initialized according to the given configuration.
func New[T mat.DType](config Config) *Model[T] {
	if config.Mask != nil && config.InputChannels != len(config.Mask) {
		panic(fmt.Sprintf("convolution: wrong mask size; found %d, expected %d", config.InputChannels, len(config.Mask)))
	}
	var paramsSize int
	if config.DepthWise {
		if config.OutputChannels != config.InputChannels {
			panic(fmt.Sprint("convolution: DepthWise convolution input channels must be equals to output channels"))
		}
		paramsSize = config.OutputChannels
	} else {
		paramsSize = config.InputChannels * config.OutputChannels
	}

	kernels := make([]nn.Param[T], paramsSize, paramsSize)
	biases := make([]nn.Param[T], paramsSize, paramsSize)
	for i := 0; i < paramsSize; i++ {
		requireGrad := config.Mask == nil || config.Mask[i%len(config.Mask)] == 1
		kernels[i] = nn.NewParam[T](mat.NewEmptyDense[T](config.KernelSizeX, config.KernelSizeY), nn.RequiresGrad[T](requireGrad))
		biases[i] = nn.NewParam[T](mat.NewEmptyVecDense[T](1), nn.RequiresGrad[T](requireGrad))
	}
	return &Model[T]{
		Config: config,
		K:      kernels,
		B:      biases,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	if m.Config.OutputChannels > 1 && m.Graph().ConcurrentComputations() > 1 {
		return m.fwdConcurrent(xs)
	}
	return m.fwdSerial(xs)
}

func (m *Model[T]) fwdSerial(xs []ag.Node[T]) []ag.Node[T] {
	ys := make([]ag.Node[T], m.Config.OutputChannels)
	for i := range ys {
		ys[i] = m.forward(xs, i)
	}
	return ys
}

func (m *Model[T]) fwdConcurrent(xs []ag.Node[T]) []ag.Node[T] {
	ys := make([]ag.Node[T], m.Config.OutputChannels)
	var wg sync.WaitGroup
	wg.Add(m.Config.OutputChannels)
	for i := 0; i < m.Config.OutputChannels; i++ {
		go func(i int) {
			defer wg.Done()
			ys[i] = m.forward(xs, i)
		}(i)
	}
	wg.Wait()
	return ys
}

func (m *Model[T]) forward(xs []ag.Node[T], outputChannel int) ag.Node[T] {
	offset := outputChannel * m.Config.InputChannels
	var out ag.Node[T]
	if m.Config.DepthWise {
		out = nn.Conv1D[T](m.K[outputChannel], xs[outputChannel], m.Config.YStride)
		out = ag.AddScalar[T](out, m.B[outputChannel])
	} else {
		for i := 0; i < len(xs); i++ {
			if m.Config.Mask == nil || m.Config.Mask[i] == 1 {
				out = ag.Add(out, nn.Conv1D[T](m.K[i+offset], xs[i], m.Config.YStride))
				out = ag.AddScalar[T](out, m.B[i+offset])
			}
		}
	}

	return ag.Invoke(m.Config.Activation, out)
}
