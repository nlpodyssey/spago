// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rmsprop

import (
	"encoding/gob"
	"fmt"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

// Config provides configuration settings for an RMSProp optimizer.
type Config struct {
	LR      float64
	Epsilon float64
	Decay   float64
}

// NewConfig returns a new RMSProp Config.
func NewConfig(lr, epsilon, decay float64) Config {
	return Config{
		LR:      lr,
		Epsilon: epsilon,
		Decay:   decay,
	}
}

// NewDefaultConfig returns a new Config with generically reasonable default values.
func NewDefaultConfig() Config {
	return Config{
		LR:      0.001,
		Epsilon: 1e-08,
		Decay:   0.95,
	}
}

//var _ optimizers.Strategy = &RMSProp[float32]{}

// The RMSProp method is a variant of AdaGrad where the squared sum of previous gradients is replaced with a moving average.
// References:
//
//	RMSProp: Divide the gradient by a running average of its recent magnitude
//	http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
type RMSProp[T float.DType] struct {
	Config
}

// New returns a new RMSProp optimizer, initialized according to the given configuration.
func New[T float.DType](c Config) *RMSProp[T] {
	return &RMSProp[T]{Config: c}
}

type State struct {
	V mat.Matrix // first moment vector
}

func init() {
	gob.Register(&State{})
}

func (o *RMSProp[T]) newState(shape ...int) *State {
	return &State{
		V: mat.NewDense[T](mat.WithShape(shape...)),
	}
}

func (o *RMSProp[T]) calculateParamUpdate(grads mat.Matrix, state *State) mat.Matrix {
	state.V.ProdScalarInPlace(o.Decay)
	buf := grads.Prod(grads)
	buf.ProdScalarInPlace(1.0 - o.Decay)
	state.V.AddInPlace(buf)
	buf2 := state.V.Sqrt()
	buf2.AddScalarInPlace(o.Epsilon)
	delta := grads.Div(buf2)
	delta.ProdScalarInPlace(o.LR)
	return delta
}

func (o *RMSProp[T]) OptimizeParams(param *nn.Param) error {
	if param.State == nil {
		param.State = o.newState(param.Value().Shape()...)
	}

	state, ok := param.State.(*State)
	if !ok {
		return fmt.Errorf("unsupported state type: %T, expected %T", param.State, &State{})
	}

	param.SubInPlace(o.calculateParamUpdate(param.Grad(), state))
	param.ZeroGrad()

	return nil
}
