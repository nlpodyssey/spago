// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package srnn implements the SRNN (Shuffling Recurrent Neural Networks) by Rotman and Wolf, 2020.
// (https://arxiv.org/pdf/2007.07324.pdf)
package srnn

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"sync"
)

var (
	_ nn.Model = &Model{}
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	Config Config
	FC     *stack.Model
	FC2    *linear.Model
	FC3    *linear.Model
	States []*State `spago:"scope:processor"`
}

// Config provides configuration settings for a SRNN Model.
type Config struct {
	InputSize  int
	HiddenSize int
	NumLayers  int
	HyperSize  int
	OutputSize int
	MultiHead  bool
}

// State represent a state of the SRNN recurrent network.
type State struct {
	Y ag.Node
	H ag.Node
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New(config Config) *Model {
	layers := []nn.StandardModel{
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

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	b := m.transformInputConcurrent(xs)
	h, _ := m.getPrevHY()
	for i := range xs {
		h, y := m.forward(h, b[i])
		m.States = append(m.States, &State{Y: y, H: h})
		ys[i] = y
	}
	return ys
}

func (m *Model) getPrevHY() (ag.Node, ag.Node) {
	if len(m.States) == 0 {
		return nil, nil
	}
	s := m.States[len(m.States)-1]
	return s.H, s.Y
}

func (m *Model) forward(hPrev, b ag.Node) (h ag.Node, y ag.Node) {
	g := m.Graph()
	if hPrev != nil {
		h = g.ReLU(g.Add(b, g.RotateR(hPrev, 1)))
	} else {
		h = g.ReLU(b)
	}
	y = nn.ToNode(m.FC3.Forward(h))
	return
}

func (m *Model) transformInput(x ag.Node) ag.Node {
	g := m.Graph()
	b := nn.ToNode(m.FC.Forward(x))
	if m.Config.MultiHead {
		sigAlphas := g.Sigmoid(nn.ToNode(m.FC2.Forward(x)))
		b = g.Prod(b, sigAlphas)
	}
	return b
}

func (m *Model) transformInputConcurrent(xs []ag.Node) []ag.Node {
	var wg sync.WaitGroup
	n := len(xs)
	wg.Add(n)
	ys := make([]ag.Node, n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			ys[i] = m.transformInput(xs[i])
		}(i)
	}
	wg.Wait()
	return ys
}
