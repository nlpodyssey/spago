// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package srnn implements the SRNN (Shuffling Recurrent Neural Networks) by Rotman and Wolf, 2020.
// (https://arxiv.org/pdf/2007.07324.pdf)
package srnn

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"sync"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the serializable parameters.
type Model struct {
	Config Config
	FC     *stack.Model
	FC2    *linear.Model
	FC3    *linear.Model
}

type Config struct {
	InputSize  int
	HiddenSize int
	NumLayers  int
	HyperSize  int
	OutputSize int
	MultiHead  bool
}

// New returns a new model with parameters initialized to zeros.
func New(config Config) *Model {
	layers := []nn.Model{
		linear.New(config.InputSize, config.HyperSize),
		layernorm.New(config.HyperSize),
		activation.New(ag.OpReLU),
	}
	for i := 1; i < config.NumLayers; i++ {
		layers = append(layers,
			linear.New(config.HyperSize, config.HyperSize),
			layernorm.New(config.HyperSize),
			activation.New(ag.OpReLU),
		)
	}
	layers = append(layers, linear.New(config.HyperSize, config.HiddenSize))
	return &Model{
		Config: config,
		FC:     stack.New(layers...),
		FC2:    linear.New(config.InputSize, config.HiddenSize),
		FC3:    linear.New(config.HiddenSize, config.OutputSize),
	}
}

type Processor struct {
	nn.BaseProcessor
	fc        *stack.Processor
	fc2       *linear.Processor
	fc3       *linear.Processor
	multiHead bool
	States    []*State
}

type State struct {
	Y ag.Node
	H ag.Node
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: false,
		},
		States:    nil,
		multiHead: m.Config.MultiHead,
		fc:        m.FC.NewProc(ctx).(*stack.Processor),
		fc2:       m.FC2.NewProc(ctx).(*linear.Processor),
		fc3:       m.FC3.NewProc(ctx).(*linear.Processor),
	}
}

// Forward performs the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	b := p.transformInputConcurrent(xs)
	h, y := p.getPrevHY()
	for i := range xs {
		h, y = p.forward(h, b[i])
		p.States = append(p.States, &State{Y: y, H: h})
		ys[i] = y
	}
	return ys
}

func (p *Processor) getPrevHY() (ag.Node, ag.Node) {
	if len(p.States) == 0 {
		return nil, nil
	}
	s := p.States[len(p.States)-1]
	return s.H, s.Y
}

func (p *Processor) forward(hPrev, b ag.Node) (h ag.Node, y ag.Node) {
	g := p.Graph
	if hPrev != nil {
		h = g.ReLU(g.Add(b, g.RotateR(hPrev, 1)))
	} else {
		h = g.ReLU(b)
	}
	y = p.fc3.Forward(h)[0]
	return
}

func (p *Processor) transformInput(x ag.Node) ag.Node {
	g := p.Graph
	b := p.fc.Forward(x)[0]
	if p.multiHead {
		sigAlphas := g.Sigmoid(p.fc2.Forward(x)[0])
		b = g.Prod(b, sigAlphas)
	}
	return b
}

func (p *Processor) transformInputConcurrent(xs []ag.Node) []ag.Node {
	var wg sync.WaitGroup
	n := len(xs)
	wg.Add(n)
	ys := make([]ag.Node, n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			ys[i] = p.transformInput(xs[i])
		}(i)
	}
	wg.Wait()
	return ys
}
