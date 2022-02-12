// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package srnn implements the SRNN (Shuffling Recurrent Neural Networks) by Rotman and Wolf, 2020.
// (https://arxiv.org/pdf/2007.07324.pdf)
package srnn

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/linear"
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/nn/stack"
	"sync"
)

var _ nn.Model[float32] = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Config Config
	FC     *stack.Model[T]
	FC2    *linear.Model[T]
	FC3    *linear.Model[T]
	States []*State[T] `spago:"scope:processor"`
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
type State[T mat.DType] struct {
	Y ag.Node[T]
	H ag.Node[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](config Config) *Model[T] {
	layers := []nn.StandardModel[T]{
		linear.New[T](config.InputSize, config.HyperSize),
		layernorm.New[T](config.HyperSize),
		activation.New[T](ag.OpReLU),
	}
	for i := 1; i < config.NumLayers; i++ {
		layers = append(layers,
			linear.New[T](config.HyperSize, config.HyperSize),
			layernorm.New[T](config.HyperSize),
			activation.New[T](ag.OpReLU),
		)
	}
	layers = append(layers, linear.New[T](config.HyperSize, config.HiddenSize))
	return &Model[T]{
		Config: config,
		FC:     stack.New[T](layers...),
		FC2:    linear.New[T](config.InputSize, config.HiddenSize),
		FC3:    linear.New[T](config.HiddenSize, config.OutputSize),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	ys := make([]ag.Node[T], len(xs))
	b := m.transformInputConcurrent(xs)
	h, _ := m.getPrevHY()
	for i := range xs {
		h, y := m.forward(h, b[i])
		m.States = append(m.States, &State[T]{Y: y, H: h})
		ys[i] = y
	}
	return ys
}

func (m *Model[T]) getPrevHY() (ag.Node[T], ag.Node[T]) {
	if len(m.States) == 0 {
		return nil, nil
	}
	s := m.States[len(m.States)-1]
	return s.H, s.Y
}

func (m *Model[T]) forward(hPrev, b ag.Node[T]) (h ag.Node[T], y ag.Node[T]) {
	g := m.Graph()
	if hPrev != nil {
		h = g.ReLU(g.Add(b, g.RotateR(hPrev, 1)))
	} else {
		h = g.ReLU(b)
	}
	y = nn.ToNode[T](m.FC3.Forward(h))
	return
}

func (m *Model[T]) transformInput(x ag.Node[T]) ag.Node[T] {
	g := m.Graph()
	b := nn.ToNode[T](m.FC.Forward(x))
	if m.Config.MultiHead {
		sigAlphas := g.Sigmoid(nn.ToNode[T](m.FC2.Forward(x)))
		b = g.Prod(b, sigAlphas)
	}
	return b
}

func (m *Model[T]) transformInputConcurrent(xs []ag.Node[T]) []ag.Node[T] {
	var wg sync.WaitGroup
	n := len(xs)
	wg.Add(n)
	ys := make([]ag.Node[T], n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			ys[i] = m.transformInput(xs[i])
		}(i)
	}
	wg.Wait()
	return ys
}
