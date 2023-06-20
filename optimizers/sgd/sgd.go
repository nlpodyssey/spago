// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sgd

import (
	"encoding/gob"
	"fmt"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

// Config provides configuration settings for an SGD optimizer.
type Config struct {
	LR       float64
	Mu       float64
	Nesterov bool
}

// NewConfig returns a new SGD Config.
func NewConfig(lr, momentum float64, nesterov bool) Config {
	return Config{
		LR:       lr,
		Mu:       momentum,
		Nesterov: nesterov,
	}
}

//var _ optimizers.Strategy = &SGD[float32]{}

// SGD implements the SGD gradient descent optimization method.
type SGD[T float.DType] struct {
	Config
	Alpha float64
}

// New returns a new SGD optimizer, initialized according to the given configuration.
func New[T float.DType](c Config) *SGD[T] {
	return &SGD[T]{Config: c, Alpha: c.LR}
}

type State struct {
	V     mat.Matrix // velocity
	Buf   mat.Matrix // buffer
	VPrev mat.Matrix // previous velocity
	VTmp  mat.Matrix // temporary velocity
}

func init() {
	gob.Register(&State{})
}

func (o *SGD[T]) newStateFor(param mat.Matrix) *State {
	shape := param.Shape()
	r, c := shape[0], shape[1]

	if o.Mu == 0.0 {
		return &State{V: param.NewMatrix(mat.WithShape(r, c))}
	}

	if !o.Nesterov {
		return &State{
			V:   param.NewMatrix(mat.WithShape(r, c)),
			Buf: param.NewMatrix(mat.WithShape(r, c)),
		}
	}

	return &State{
		V:     param.NewMatrix(mat.WithShape(r, c)),
		Buf:   param.NewMatrix(mat.WithShape(r, c)),
		VPrev: param.NewMatrix(mat.WithShape(r, c)),
		VTmp:  param.NewMatrix(mat.WithShape(r, c)),
	}
}

func (o *SGD[T]) calculateParamUpdate(grads mat.Matrix, state *State) mat.Matrix {
	if o.Mu == 0.0 {
		return o.calculateParamUpdateSGD(grads, state)
	}
	if o.Nesterov {
		return o.calculateParamUpdateNesterovMomentum(grads, state)
	}
	return o.calculateParamUpdateMomentum(grads, state)

}

func (o *SGD[T]) calculateParamUpdateSGD(grads mat.Matrix, state *State) mat.Matrix {
	return state.V.ProdMatrixScalarInPlace(grads, o.Alpha)
}

func (o *SGD[T]) calculateParamUpdateMomentum(grads mat.Matrix, state *State) mat.Matrix {
	state.Buf.ProdMatrixScalarInPlace(grads, o.Alpha)
	state.V.ProdScalarInPlace(o.Mu)
	state.V.AddInPlace(state.Buf)
	return state.V
}

func (o *SGD[T]) calculateParamUpdateNesterovMomentum(grads mat.Matrix, state *State) mat.Matrix {
	state.Buf.ProdMatrixScalarInPlace(grads, o.Alpha)
	state.VPrev.ProdMatrixScalarInPlace(state.V, o.Mu)
	state.V.ProdScalarInPlace(o.Mu)
	state.V.AddInPlace(state.Buf) // += grad * alpha
	state.VTmp.ProdMatrixScalarInPlace(state.V, 1.0+o.Mu)
	state.VTmp.SubInPlace(state.VPrev)
	return state.VTmp
}

func (o *SGD[T]) OptimizeParams(param *nn.Param) error {
	if param.State == nil {
		param.State = o.newStateFor(param)
	}

	state, ok := param.State.(*State)
	if !ok {
		return fmt.Errorf("unsupported state type: %T, expected %T", param.State, &State{})
	}

	param.SubInPlace(o.calculateParamUpdate(param.Grad().(mat.Matrix), state))
	param.ZeroGrad()

	return nil
}
