// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// srnn implements the SRNN (Shuffling Recurrent Neural Networks) by Rotman and Wolf, 2020.
// (https://arxiv.org/pdf/2007.07324.pdf)
package srnn

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
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
	layers := []nn.Model{linear.New(config.InputSize, config.HyperSize), activation.New(ag.OpReLU)}
	for i := 1; i < config.NumLayers; i++ {
		layers = append(layers, linear.New(config.HyperSize, config.HyperSize), activation.New(ag.OpReLU))
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
	fc     *stack.Processor
	fc2    *linear.Processor
	fc3    *linear.Processor
	States []*State
}

type State struct {
	Y ag.Node
	H ag.Node
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(g *ag.Graph) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: false,
		},
		States: nil,
		fc:     m.FC.NewProc(g).(*stack.Processor),
		fc2:    m.FC2.NewProc(g).(*linear.Processor),
		fc3:    m.FC3.NewProc(g).(*linear.Processor),
	}
}

// Forward performs the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	h, y := p.getPrevHY()
	for i, x := range xs {
		h, y = p.forward(h, x)
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

func (p *Processor) forward(hPrev, x ag.Node) (h ag.Node, y ag.Node) {
	b := p.fc.Forward(x)[0]
	if p.Model.(*Model).Config.MultiHead {
		sigAlphas := p.Graph.Sigmoid(p.fc2.Forward(x)[0])
		b = p.Graph.Prod(b, sigAlphas)
	}
	if hPrev != nil {
		h = p.Graph.ReLU(p.Graph.Add(b, p.Graph.RotateR(hPrev, 1)))
	} else {
		h = p.Graph.ReLU(b)
	}
	y = p.fc3.Forward(h)[0]
	return
}
