// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"github.com/nlpodyssey/spago/pkg/mat/f64utils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"math"
)

var (
	_ nn.Model     = &Discriminator{}
	_ nn.Processor = &DiscriminatorProcessor{}
)

type DiscriminatorConfig struct {
	InputSize        int
	HiddenSize       int
	HiddenActivation ag.OpName
	OutputActivation ag.OpName
}

type Discriminator struct {
	*stack.Model
}

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

type DiscriminatorProcessor struct {
	*stack.Processor
}

func (m *Discriminator) NewProc(ctx nn.Context) nn.Processor {
	return &DiscriminatorProcessor{
		Processor: m.Model.NewProc(ctx).(*stack.Processor),
	}
}

func (p *DiscriminatorProcessor) Discriminate(encoded []ag.Node) []int {
	ys := make([]int, len(encoded))
	for i, x := range p.Processor.Forward(encoded...) {
		ys[i] = int(math.Round(float64(f64utils.Sign(x.ScalarValue())+1.0) / 2.0))
	}
	return ys
}
