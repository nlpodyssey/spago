// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package srnn implements the SRNN (Shuffling Recurrent Neural Networks) by Rotman and Wolf, 2020.
// (https://arxiv.org/pdf/2007.07324.pdf)
package srnn

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/linear"
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/nn/stack"
)

var _ nn.Model = &Model{}

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	Config Config
	FC     *stack.Model
	FC2    *linear.Model
	FC3    *linear.Model
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
// TODO: is this used?
type State struct {
	Y ag.Node
	H ag.Node
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T float.DType](config Config) *Model {
	layers := []nn.StandardModel{
		linear.New[T](config.InputSize, config.HyperSize),
		layernorm.New[T](config.HyperSize, 1e-5),
		activation.New(activation.ReLU),
	}
	for i := 1; i < config.NumLayers; i++ {
		layers = append(layers,
			linear.New[T](config.HyperSize, config.HyperSize),
			layernorm.New[T](config.HyperSize, 1e-5),
			activation.New(activation.ReLU),
		)
	}
	layers = append(layers, linear.New[T](config.HyperSize, config.HiddenSize))
	return &Model{
		Config: config,
		FC:     stack.New(layers...),
		FC2:    linear.New[T](config.InputSize, config.HiddenSize),
		FC3:    linear.New[T](config.HiddenSize, config.OutputSize),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	b := m.transformInput(xs)
	var h ag.Node = nil
	for i := range xs {
		h, ys[i] = m.Next(h, b[i])
	}
	return ys
}

// Next performs a single forward step, producing a new state.
func (m *Model) Next(hPrev, b ag.Node) (h ag.Node, y ag.Node) {
	if hPrev != nil {
		h = ag.ReLU(ag.Add(b, ag.RotateR(hPrev, 1)))
	} else {
		h = ag.ReLU(b)
	}
	y = m.FC3.Forward(h)[0]
	return
}

func (m *Model) transformInput(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
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
