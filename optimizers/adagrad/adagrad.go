// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package adagrad

import (
	"encoding/gob"
	"fmt"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

// Config provides configuration settings for an AdaGrad optimizer.
type Config struct {
	LR      float64
	Epsilon float64
}

// NewConfig returns a new AdaGrad Config.
func NewConfig(lr, epsilon float64) Config {
	return Config{
		LR:      lr,
		Epsilon: epsilon,
	}
}

type State struct {
	M mat.Matrix // sum of squares of historical gradients
}

func init() {
	gob.Register(&State{})
}

// NewDefaultConfig returns a new Config with generically reasonable default values.
func NewDefaultConfig() Config {
	return Config{
		LR:      0.01,
		Epsilon: 1.0e-8,
	}
}

// AdaGrad assigns a different learning rate to each parameter using the sum of squares of its all historical gradients.
// References
//
//	Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
//	http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
type AdaGrad[T float.DType] struct {
	Config
}

// New returns a new AdaGrad optimizer, initialized according to the given configuration.
func New[T float.DType](c Config) *AdaGrad[T] {
	return &AdaGrad[T]{
		Config: c,
	}
}

func (o *AdaGrad[T]) newState(shape ...int) *State {
	return &State{
		M: mat.NewDense[T](mat.WithShape(shape...)),
	}
}

// m = m + grads*grads
// delta = (grads / (sqrt(m) + eps)) * lr
func (o *AdaGrad[T]) calculateParamUpdate(grads mat.Matrix, state *State) mat.Matrix {
	state.M.AddInPlace(grads.Prod(grads))
	return grads.Div(state.M.Sqrt().AddScalarInPlace(o.Epsilon)).ProdScalarInPlace(o.LR)
}

func (o *AdaGrad[T]) OptimizeParams(param *nn.Param) error {
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
