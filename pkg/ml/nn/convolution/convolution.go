// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package convolution

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"log"
	"sync"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

type Model struct {
	K              []*nn.Param `type:"weights"`
	B              []*nn.Param `type:"biases"`
	Activation     ag.OpName   // output activation
	inputChannels  int
	outputChannels int
	xStride        int
	yStride        int
}

func New(kernelSizeX, kernelSizeY, xStride, yStride, inputChannels, outputChannels int, activation ag.OpName) *Model {
	paramsSize := inputChannels * outputChannels
	kernels := make([]*nn.Param, paramsSize, paramsSize)
	biases := make([]*nn.Param, paramsSize, paramsSize)
	for i := 0; i < paramsSize; i++ {
		kernels[i] = nn.NewParam(mat.NewEmptyDense(kernelSizeX, kernelSizeY))
		biases[i] = nn.NewParam(mat.NewEmptyVecDense(1))
	}
	return &Model{
		K:              kernels,
		B:              biases,
		Activation:     activation,
		inputChannels:  inputChannels,
		outputChannels: outputChannels,
		xStride:        xStride,
		yStride:        yStride,
	}
}

// SetActivation sets the new activation and returns the previous one.
func (m *Model) SetActivation(a ag.OpName) ag.OpName {
	prev := m.Activation
	m.Activation = a
	return prev
}

type Processor struct {
	opt                     []interface{}
	model                   *Model
	mode                    nn.ProcessingMode
	g                       *ag.Graph
	k                       []ag.Node
	b                       []ag.Node
	ConcurrentOutputChannel bool
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		model:                   m,
		mode:                    nn.Training,
		opt:                     opt,
		g:                       g,
		k:                       nn.AttachParamsToGraph(g, m.K...),
		b:                       nn.AttachParamsToGraph(g, m.B...),
		ConcurrentOutputChannel: true,
	}
	p.init(opt)
	return p
}

func (p *Processor) Model() nn.Model                { return p.model }
func (p *Processor) Graph() *ag.Graph               { return p.g }
func (p *Processor) RequiresFullSeq() bool          { return true }
func (p *Processor) Mode() nn.ProcessingMode        { return p.mode }
func (p *Processor) SetMode(mode nn.ProcessingMode) { p.mode = mode }

type Concurrency struct {
	Value bool
}

func (p *Processor) init(opt []interface{}) {
	for _, t := range opt {
		switch t := t.(type) {
		case Concurrency:
			p.ConcurrentOutputChannel = t.Value
		default:
			log.Fatal("convolution: invalid init options")
		}
	}
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	if p.ConcurrentOutputChannel && p.model.outputChannels > 1 {
		return p.fwdConcurrent(xs)
	} else {
		return p.fwdSerial(xs)
	}
}

func (p *Processor) fwdSerial(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, p.model.outputChannels)
	for i := range ys {
		ys[i] = p.forward(xs, i)
	}
	return ys
}

func (p *Processor) fwdConcurrent(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, p.model.outputChannels)
	var wg sync.WaitGroup
	wg.Add(p.model.outputChannels)
	for i := 0; i < p.model.outputChannels; i++ {
		go func(i int) {
			defer wg.Done()
			ys[i] = p.forward(xs, i)
		}(i)
	}
	wg.Wait()
	return ys
}

func (p *Processor) forward(xs []ag.Node, outputChannel int) ag.Node {
	offset := outputChannel * p.model.inputChannels
	out := nn.Conv2D(p.g, p.k[0+offset], xs[0], p.model.xStride, p.model.yStride)
	out = p.g.AddScalar(out, p.b[0+offset])
	for i := 1; i < len(xs); i++ {
		out = p.g.Add(out, nn.Conv2D(p.g, p.k[i+offset], xs[i], p.model.xStride, p.model.yStride))
		out = p.g.AddScalar(out, p.b[i+offset])
	}
	return p.g.Invoke(p.model.Activation, out)
}
