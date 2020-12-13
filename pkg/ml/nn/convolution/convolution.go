// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package convolution

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"sync"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

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

type Model struct {
	Config
	K []*nn.Param `type:"weights"`
	B []*nn.Param `type:"biases"`
}

func New(config Config) *Model {
	if config.Mask != nil && config.InputChannels != len(config.Mask) {
		panic(fmt.Sprintf("convolution: wrong mask size; found %d, expected %d", config.InputChannels, len(config.Mask)))
	}
	paramsSize := config.InputChannels * config.OutputChannels
	kernels := make([]*nn.Param, paramsSize, paramsSize)
	biases := make([]*nn.Param, paramsSize, paramsSize)
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

type Processor struct {
	nn.BaseProcessor
	Config
	k []ag.Node
	b []ag.Node
	// whether to enable the concurrent forward computation on the output channel
	concurrent bool
}

func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	k := make([]ag.Node, len(m.K))
	b := make([]ag.Node, len(m.B))
	for i := range m.K {
		k[i] = ctx.Graph.NewWrap(m.K[i])
		b[i] = ctx.Graph.NewWrap(m.B[i])
	}
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		Config:     m.Config,
		k:          k,
		b:          b,
		concurrent: true,
	}
}

func (p *Processor) SetConcurrentComputations(value bool) {
	p.concurrent = value
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	if p.concurrent && p.OutputChannels > 1 {
		return p.fwdConcurrent(xs)
	}
	return p.fwdSerial(xs)
}

func (p *Processor) fwdSerial(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, p.OutputChannels)
	for i := range ys {
		ys[i] = p.forward(xs, i)
	}
	return ys
}

func (p *Processor) fwdConcurrent(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, p.OutputChannels)
	var wg sync.WaitGroup
	wg.Add(p.OutputChannels)
	for i := 0; i < p.OutputChannels; i++ {
		go func(i int) {
			defer wg.Done()
			ys[i] = p.forward(xs, i)
		}(i)
	}
	wg.Wait()
	return ys
}

func (p *Processor) forward(xs []ag.Node, outputChannel int) ag.Node {
	g := p.Graph
	offset := outputChannel * p.InputChannels
	var out ag.Node
	for i := 0; i < len(xs); i++ {
		if p.Config.Mask == nil || p.Config.Mask[i] == 1 {
			out = g.Add(out, nn.Conv2D(g, p.k[i+offset], xs[i], p.XStride, p.YStride))
			out = g.AddScalar(out, p.b[i+offset])
		}
	}
	return g.Invoke(p.Activation, out)
}
