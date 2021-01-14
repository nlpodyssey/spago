// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/gob"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/floatutils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

var (
	_ nn.Model = &Discriminator{}
)

// DiscriminatorConfig provides configuration settings for a BERT Discriminator.
type DiscriminatorConfig struct {
	InputSize        int
	HiddenSize       int
	HiddenActivation ag.OpName
	OutputActivation ag.OpName
}

// Discriminator is a BERT Discriminator model.
type Discriminator struct {
	*stack.Model
}

func init() {
	gob.Register(&Discriminator{})
}

// NewDiscriminator returns a new BERT Discriminator model.
func NewDiscriminator(config DiscriminatorConfig) *Discriminator {
	return &Discriminator{
		Model: stack.New(
			linear.New(config.InputSize, config.HiddenSize),
			activation.New(config.HiddenActivation),
			linear.New(config.HiddenSize, 1),
			activation.New(config.OutputActivation),
		),
	}
}

// Discriminate returns 0 or 1 for each encoded element, where 1 means that
// the word is out of context.
func (m *Discriminator) Discriminate(encoded []ag.Node) []int {
	ys := make([]int, len(encoded))
	for i, x := range m.Forward(encoded...) {
		ys[i] = int(mat.Round(mat.Float(floatutils.Sign(x.ScalarValue())+1.0) / 2.0))
	}
	return ys
}
