// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package srnn implements the SRNN (Shuffling Recurrent Neural Networks) by Rotman and Wolf, 2020.
// (https://arxiv.org/pdf/2007.07324.pdf)
//
// This file implements a bidirectional variant of the SRNN, in which the input features are shared
// among the two directions (Grella, 2021).
package srnn

import (
	"encoding/gob"
	"sync"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/linear"
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/nn/stack"
)

var _ nn.Model = &BiModel[float32]{}

// BiModel contains the serializable parameters.
type BiModel[T mat.DType] struct {
	nn.Module
	Config    Config
	FC        *stack.Model[T]
	FC2       *linear.Model[T]
	FC3       *linear.Model[T]
	LayerNorm *layernorm.Model[T]
}

func init() {
	gob.Register(&BiModel[float32]{})
	gob.Register(&BiModel[float64]{})
}

// NewBidirectional returns a new model with parameters initialized to zeros.
func NewBidirectional[T mat.DType](config Config) *BiModel[T] {
	layers := []nn.StandardModel[T]{
		linear.New[T](config.InputSize, config.HyperSize),
		activation.New[T](activation.ReLU),
	}
	for i := 1; i < config.NumLayers; i++ {
		layers = append(layers,
			linear.New[T](config.HyperSize, config.HyperSize),
			activation.New[T](activation.ReLU),
		)
	}
	layers = append(layers, linear.New[T](config.HyperSize, config.HiddenSize))
	return &BiModel[T]{
		Config:    config,
		FC:        stack.New(layers...),
		FC2:       linear.New[T](config.InputSize, config.HiddenSize),
		FC3:       linear.New[T](config.HiddenSize*2, config.OutputSize),
		LayerNorm: layernorm.New[T](config.OutputSize, 1e-5),
	}
}

// Forward performs the forward step for each input and returns the result.
func (m *BiModel[T]) Forward(xs ...ag.Node) []ag.Node {
	n := len(xs)
	ys := make([]ag.Node, n)
	b := m.transformInputConcurrent(xs)

	var hfwd []ag.Node
	var hbwd []ag.Node
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		hfwd = m.forwardHidden(b)
	}()
	go func() {
		defer wg.Done()
		hbwd = m.forwardHidden(reversed(b))
	}()
	wg.Wait()

	for i := 0; i < n; i++ {
		concat := ag.Concat(hfwd[i], hbwd[n-1-i])
		ys[i] = m.FC3.Forward(concat)[0]
	}
	ys = m.LayerNorm.Forward(ys...)
	return ys
}

func (m *BiModel[T]) forwardHidden(b []ag.Node) []ag.Node {
	n := len(b)
	h := make([]ag.Node, n)
	h[0] = ag.ReLU(b[0])
	for i := 1; i < n; i++ {
		h[i] = ag.ReLU(ag.Add(b[i], ag.RotateR(h[i-1], 1)))
	}
	return h
}

func (m *BiModel[T]) transformInput(x ag.Node) ag.Node {
	b := m.FC.Forward(x)[0]
	if m.Config.MultiHead {
		sigAlphas := ag.Sigmoid(m.FC2.Forward(x)[0])
		b = ag.Prod(b, sigAlphas)
	}
	return b
}

func (m *BiModel[T]) transformInputConcurrent(xs []ag.Node) []ag.Node {
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

func reversed(ns []ag.Node) []ag.Node {
	r := make([]ag.Node, len(ns))
	copy(r, ns)
	for i := 0; i < len(r)/2; i++ {
		j := len(r) - i - 1
		r[i], r[j] = r[j], r[i]
	}
	return r
}
