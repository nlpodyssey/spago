// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

var (
	_ nn.Model[float32] = &Discriminator[float32]{}
)

// DiscriminatorConfig provides configuration settings for a BERT Discriminator.
type DiscriminatorConfig struct {
	InputSize        int
	HiddenSize       int
	HiddenActivation ag.OpName
	OutputActivation ag.OpName
}

// Discriminator is a BERT Discriminator model.
type Discriminator[T mat.DType] struct {
	*stack.Model[T]
}

func init() {
	gob.Register(&Discriminator[float32]{})
	gob.Register(&Discriminator[float64]{})
}

// NewDiscriminator returns a new BERT Discriminator model.
func NewDiscriminator[T mat.DType](config DiscriminatorConfig) *Discriminator[T] {
	return &Discriminator[T]{
		Model: stack.New[T](
			linear.New[T](config.InputSize, config.HiddenSize),
			activation.New[T](config.HiddenActivation),
			linear.New[T](config.HiddenSize, 1),
			activation.New[T](config.OutputActivation),
		),
	}
}

// Discriminate returns 0 or 1 for each encoded element, where 1 means that
// the word is out of context.
func (m *Discriminator[T]) Discriminate(encoded []ag.Node[T]) []int {
	ys := make([]int, len(encoded)) // all zeros by default
	for i, x := range m.Forward(encoded...) {
		if x.ScalarValue() >= 0 {
			ys[i] = 1
		}
	}
	return ys
}
