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
)

var _ nn.Model[float32] = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Config Config
	FC     *stack.Model[T]
	FC2    *linear.Model[T]
	FC3    *linear.Model[T]
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
		activation.New[T](activation.ReLU),
	}
	for i := 1; i < config.NumLayers; i++ {
		layers = append(layers,
			linear.New[T](config.HyperSize, config.HyperSize),
			layernorm.New[T](config.HyperSize),
			activation.New[T](activation.ReLU),
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
	b := m.transformInput(xs)
	var h ag.Node[T] = nil
	for i := range xs {
		h, ys[i] = m.Next(h, b[i])
	}
	return ys
}

// Next performs a single forward step, producing a new state.
func (m *Model[T]) Next(hPrev, b ag.Node[T]) (h ag.Node[T], y ag.Node[T]) {
	if hPrev != nil {
		h = ag.ReLU(ag.Add(b, ag.RotateR(hPrev, 1)))
	} else {
		h = ag.ReLU(b)
	}
	y = m.FC3.Forward(h)[0]
	return
}

func (m *Model[T]) transformInput(xs []ag.Node[T]) []ag.Node[T] {
	ys := make([]ag.Node[T], len(xs))
	for i, x := range xs {
		b := m.FC.Forward(x)[0]
		if m.Config.MultiHead {
			sigAlphas := ag.Sigmoid(m.FC2.Forward(x)[0])
			b = ag.Prod(b, sigAlphas)
		}
		ys[i] = b
	}
	return ys
}
