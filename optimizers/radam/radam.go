// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package radam

import (
	"encoding/gob"
	"fmt"
	"math"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

// Config provides configuration settings for a RAdam optimizer.
type Config struct {
	StepSize float64
	Beta1    float64
	Beta2    float64
	Epsilon  float64
}

// NewConfig returns a new RAdam Config.
// It panics if beta1 or beta2 are not in the range [0.0, 1.0).
func NewConfig(stepSize, beta1, beta2, epsilon float64) Config {
	if !(beta1 >= 0.0 && beta1 < 1.0) {
		panic("adam: `beta1` must be in the range [0.0, 1.0)")
	}
	if !(beta2 >= 0.0 && beta2 < 1.0) {
		panic("adam: `beta2` must be in the range [0.0, 1.0)")
	}
	return Config{
		StepSize: stepSize,
		Beta1:    beta1,
		Beta2:    beta2,
		Epsilon:  epsilon,
	}
}

// NewDefaultConfig returns a new Config with generically reasonable default values.
func NewDefaultConfig() Config {
	return Config{
		StepSize: 0.001,
		Beta1:    0.9,
		Beta2:    0.999,
		Epsilon:  1.0e-8,
	}
}

// RAdam implements the RAdam gradient descent optimization method.
type RAdam[T float.DType] struct {
	Config
	RoMax    float64 // The maximum length of the approximated SMA.
	TimeStep int
}

// New returns a new RAdam optimizer, initialized according to the given configuration.
func New[T float.DType](c Config) *RAdam[T] {
	adam := &RAdam[T]{
		Config:   c,
		RoMax:    2.0/(1.0-c.Beta2) - 1.0,
		TimeStep: 1.0,
	}
	return adam
}

type State struct {
	M    mat.Matrix // first moment vector
	V    mat.Matrix // second moment vector
	Buf1 mat.Matrix // buffer for first moment vector
	Buf2 mat.Matrix // buffer for second moment vector
	Buf3 mat.Matrix // buffer for second moment vector
}

func init() {
	gob.Register(&State{})
}

// newState returns a new state.
func (o *RAdam[T]) newState(shape ...int) *State {
	r, c := shape[0], shape[1]
	return &State{
		M:    mat.NewDense[T](mat.WithShape(r, c)),
		V:    mat.NewDense[T](mat.WithShape(r, c)),
		Buf1: mat.NewDense[T](mat.WithShape(r, c)),
		Buf2: mat.NewDense[T](mat.WithShape(r, c)),
		Buf3: mat.NewDense[T](mat.WithShape(r, c)),
	}
}

// IncBatch beats the occurrence of a new batch.
func (o *RAdam[_]) IncBatch() {
	o.TimeStep++
}

func (o *RAdam[T]) calculateParamUpdate(grads mat.Matrix, state *State) mat.Matrix {
	updateM(grads, state, o.Beta1)
	updateV(grads, state, o.Beta2)
	sqrtB2T := math.Sqrt(1.0 - math.Pow(o.Beta2, float64(o.TimeStep)))
	alpha := o.calcAlpha()
	buf := state.V.Sqrt().AddScalarInPlace(o.Epsilon * sqrtB2T)
	suppDiv := state.M.Div(buf)
	state.Buf3.ProdMatrixScalarInPlace(suppDiv, alpha)
	return state.Buf3
}

// m = m*beta1 + grads*(1.0-beta1)
func updateM(grads mat.Matrix, state *State, beta1 float64) {
	state.M.ProdScalarInPlace(beta1)
	state.Buf1.ProdMatrixScalarInPlace(grads, 1.0-beta1)
	state.M.AddInPlace(state.Buf1)
}

// v = v*beta2 + (grads*grads)*(1.0-beta2)
func updateV(grads mat.Matrix, state *State, beta2 float64) {
	state.V.ProdScalarInPlace(beta2)
	sqGrad := grads.Prod(grads)
	state.Buf2.ProdMatrixScalarInPlace(sqGrad, 1.0-beta2)
	state.V.AddInPlace(state.Buf2)
}

func (o *RAdam[T]) calcAlpha() float64 {
	timeStep := float64(o.TimeStep)
	b1T := math.Pow(o.Beta1, timeStep)
	b2T := math.Pow(o.Beta2, timeStep)
	ro := o.RoMax - 2.0*timeStep*b2T/(1.0-b2T)
	var rect float64 = 1
	if ro > 4.0 { // i.e. if the variance is tractable
		rect = math.Sqrt((ro - 4.0) * (ro - 2.0) * o.RoMax / ((o.RoMax - 4.0) * (o.RoMax - 2.0) * ro))
	}
	return o.StepSize * rect * mat.Sqrt(1.0-b2T) / (1.0 - b1T)
}

func (o *RAdam[T]) OptimizeParams(param *nn.Param) error {
	if param.State == nil {
		param.State = o.newState(param.Value().Shape()...)
	}

	state, ok := param.State.(*State)
	if !ok {
		return fmt.Errorf("unsupported state type: %T, expected %T", param.State, &State{})
	}

	param.SubInPlace(o.calculateParamUpdate(param.Grad().(mat.Matrix), state))
	param.ZeroGrad()

	return nil
}
