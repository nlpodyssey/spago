// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package convolution

import (
	"encoding/gob"
	"fmt"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"sync"
)

var (
	_ nn.Model = &Model{}
)

// Config provides configuration settings for a convolution Model.
type Config struct {
	KernelSizeX    int
	KernelSizeY    int
	XStride        int
	YStride        int
	InputChannels  int
	OutputChannels int
	Mask           []int
	Activation     ag.OpName
}

// Model contains the serializable parameters for a convolutional neural network model.
type Model struct {
	nn.BaseModel
	Config Config
	K      []nn.Param `spago:"type:weights"`
	B      []nn.Param `spago:"type:biases"`
}

func init() {
	gob.Register(&Model{})
}

// New returns a new convolution Model, initialized according to the given configuration.
func New(config Config) *Model {
	if config.Mask != nil && config.InputChannels != len(config.Mask) {
		panic(fmt.Sprintf("convolution: wrong mask size; found %d, expected %d", config.InputChannels, len(config.Mask)))
	}
	paramsSize := config.InputChannels * config.OutputChannels
	kernels := make([]nn.Param, paramsSize, paramsSize)
	biases := make([]nn.Param, paramsSize, paramsSize)
	for i := 0; i < paramsSize; i++ {
		requireGrad := config.Mask == nil || config.Mask[i%len(config.Mask)] == 1
		kernels[i] = nn.NewParam(mat.NewEmptyDense(config.KernelSizeX, config.KernelSizeY), nn.RequiresGrad(requireGrad))
		biases[i] = nn.NewParam(mat.NewEmptyVecDense(1), nn.RequiresGrad(requireGrad))
	}
	return &Model{
		Config: config,
		K:      kernels,
		B:      biases,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	if m.Config.OutputChannels > 1 && m.Graph().ConcurrentComputations() > 1 {
		return m.fwdConcurrent(xs)
	}
	return m.fwdSerial(xs)
}

func (m *Model) fwdSerial(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, m.Config.OutputChannels)
	for i := range ys {
		ys[i] = m.forward(xs, i)
	}
	return ys
}

func (m *Model) fwdConcurrent(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, m.Config.OutputChannels)
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

func (m *Model) forward(xs []ag.Node, outputChannel int) ag.Node {
	g := m.Graph()
	offset := outputChannel * m.Config.InputChannels
	var out ag.Node
	for i := 0; i < len(xs); i++ {
		if m.Config.Mask == nil || m.Config.Mask[i] == 1 {
			out = g.Add(out, nn.Conv2D(g, m.K[i+offset], xs[i], m.Config.XStride, m.Config.YStride))
			out = g.AddScalar(out, m.B[i+offset])
		}
	}
	return g.Invoke(m.Config.Activation, out)
}
