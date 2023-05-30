// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package adam

import (
	"encoding/gob"
	"fmt"
	"math"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

// Config provides configuration settings for an Adam optimizer.
type Config struct {
	StepSize float64
	Beta1    float64
	Beta2    float64
	Epsilon  float64
	Lambda   float64 // AdamW
}

// NewConfig returns a new Adam Config.
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
		Lambda:   0.0,
	}
}

// NewAdamWConfig returns a new Adam Config.
func NewAdamWConfig(stepSize, beta1, beta2, epsilon, lambda float64) Config {
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
		Lambda:   lambda,
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

//var _ optimizers.Strategy = &Adam[float32]{}

// Adam implements the Adam gradient descent optimization method.
type Adam struct {
	Config
	Alpha    float64
	TimeStep int
	adamw    bool
}

// New returns a new Adam optimizer, initialized according to the given configuration.
func New(c Config) *Adam {
	adam := &Adam{
		Config: c,
		Alpha:  c.StepSize,
		adamw:  c.Lambda != 0.0,
	}
	adam.IncExample() // initialize 'alpha' coefficient
	return adam
}

type State struct {
	V    mat.Matrix // first moment vector
	M    mat.Matrix // second raw moment vector
	Buf1 mat.Matrix // contains 'grads.ProdScalar(1.0 - beta1)'
	Buf2 mat.Matrix // contains 'grads.Prod(grads).ProdScalar(1.0 - beta2)'
	Buf3 mat.Matrix
}

func init() {
	gob.Register(&State{})
}

func (o *Adam) newStateFor(param mat.Matrix) *State {
	shape := param.Shape()
	return &State{
		V:    param.NewMatrix(mat.WithShape(shape...)),
		M:    param.NewMatrix(mat.WithShape(shape...)),
		Buf1: param.NewMatrix(mat.WithShape(shape...)),
		Buf2: param.NewMatrix(mat.WithShape(shape...)),
		Buf3: param.NewMatrix(mat.WithShape(shape...)),
	}
}

// IncExample beats the occurrence of a new example.
func (o *Adam) IncExample() {
	o.TimeStep++
	o.updateAlpha()
}

func (o *Adam) updateAlpha() {
	ts := float64(o.TimeStep)
	o.Alpha = o.StepSize * math.Sqrt(1.0-math.Pow(o.Beta2, ts)) / (1.0 - math.Pow(o.Beta1, ts))
}

// v = v*beta1 + grads*(1.0-beta1)
// m = m*beta2 + (grads*grads)*(1.0-beta2)
// d = (v / (sqrt(m) + eps)) * alpha
func (o *Adam) calculateParamUpdate(grads mat.Matrix, state *State) mat.Matrix {
	updateV(grads, state, o.Beta1)
	updateM(grads, state, o.Beta2)
	buf := state.M.Sqrt().AddScalarInPlace(o.Epsilon)
	suppDiv := state.V.Div(buf)
	state.Buf3.ProdMatrixScalarInPlace(suppDiv, o.Alpha)
	return state.Buf3
}

// v = v*beta1 + grads*(1.0-beta1)
// m = m*beta2 + (grads*grads)*(1.0-beta2)
// d = (v / (sqrt(m) + eps))  + (lambda * weights) + alpha
func (o *Adam) calculateParamUpdateW(grads mat.Matrix, state *State, weights mat.Matrix) mat.Matrix {
	updateV(grads, state, o.Beta1)
	updateM(grads, state, o.Beta2)
	buf := state.M.Sqrt().AddScalarInPlace(o.Epsilon)
	suppDiv := state.V.Div(buf)
	scaledW := weights.ProdScalar(o.Lambda)
	suppDiv.AddInPlace(scaledW)
	state.Buf3.ProdMatrixScalarInPlace(suppDiv, o.Alpha)
	return state.Buf3
}

// v = v*beta1 + grads*(1.0-beta1)
func updateV(grads mat.Matrix, state *State, beta1 float64) {
	state.V.ProdScalarInPlace(beta1)
	state.Buf1.ProdMatrixScalarInPlace(grads, 1.0-beta1)
	state.V.AddInPlace(state.Buf1)
}

// m = m*beta2 + (grads*grads)*(1.0-beta2)
func updateM(grads mat.Matrix, state *State, beta2 float64) {
	state.M.ProdScalarInPlace(beta2)
	sqGrad := grads.Prod(grads)
	state.Buf2.ProdMatrixScalarInPlace(sqGrad, 1.0-beta2)
	state.M.AddInPlace(state.Buf2)
}

func (o *Adam) OptimizeParams(param *nn.Param) error {
	if param.State == nil {
		param.State = o.newStateFor(param)
	}

	state, ok := param.State.(*State)
	if !ok {
		return fmt.Errorf("unsupported state type: %T, expected %T", param.State, &State{})
	}

	if o.adamw {
		param.SubInPlace(o.calculateParamUpdateW(param.Grad(), state, param.Value()))
		param.ZeroGrad()
		return nil
	}

	param.SubInPlace(o.calculateParamUpdate(param.Grad(), state))
	param.ZeroGrad()

	return nil
}
