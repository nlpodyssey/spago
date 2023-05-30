// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lamb

import (
	"encoding/gob"
	"fmt"
	"math"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

// Config provides configuration settings for Lamb optimizer.
type Config struct {
	StepSize float64
	Beta1    float64
	Beta2    float64
	Epsilon  float64
	Lambda   float64
}

// NewConfig returns a new Lamb Config.
func NewConfig(stepSize, beta1, beta2, epsilon, lambda float64) Config {
	if !(beta1 >= 0.0 && beta1 < 1.0) {
		panic("lamb: `beta1` must be in the range [0.0, 1.0)")
	}
	if !(beta2 >= 0.0 && beta2 < 1.0) {
		panic("lamb: `beta2` must be in the range [0.0, 1.0)")
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
		Lambda:   0.1,
	}
}

// Lamb implements the Lamb gradient descent optimization method.
type Lamb[T float.DType] struct {
	Config
	Alpha    float64
	TimeStep int
}

// New returns a new Lamb optimizer, initialized according to the given configuration.
func New[T float.DType](c Config) *Lamb[T] {
	lamb := &Lamb[T]{
		Config: c,
		Alpha:  c.StepSize,
	}
	lamb.IncExample() // initialize 'alpha' coefficient
	return lamb
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

func (o *Lamb[T]) newState(shape ...int) *State {
	r, c := shape[0], shape[1]
	return &State{
		V:    mat.NewDense[T](mat.WithShape(r, c)),
		M:    mat.NewDense[T](mat.WithShape(r, c)),
		Buf1: mat.NewDense[T](mat.WithShape(r, c)),
		Buf2: mat.NewDense[T](mat.WithShape(r, c)),
		Buf3: mat.NewDense[T](mat.WithShape(r, c)),
	}
}

// IncExample beats the occurrence of a new example.
func (o *Lamb[_]) IncExample() {
	o.TimeStep++
	o.updateAlpha()
}

func (o *Lamb[T]) updateAlpha() {
	ts := float64(o.TimeStep)
	o.Alpha = o.StepSize * math.Sqrt(1.0-math.Pow(o.Beta2, ts)) / (1.0 - math.Pow(o.Beta1, ts))
}

// CalcDelta returns the difference between the current params and where the method wants it to be.
func (o *Lamb[T]) CalcDelta(state *State, cur mat.Matrix, grads mat.Matrix) mat.Matrix {
	return o.calculateParamUpdate(grads, state, cur.Value())
}

// v = v*beta1 + grads*(1.0-beta1)
// m = m*beta2 + (grads*grads)*(1.0-beta2)
// weights = ||params|| / || (v / (sqrt(m) + eps)) + (lambda * weights)
// d = (v / (sqrt(m) + eps)) + (lambda * weights) * alpha
func (o *Lamb[T]) calculateParamUpdate(grads mat.Matrix, state *State, weights mat.Matrix) mat.Matrix {
	updateV(grads, state, o.Beta1)
	updateM(grads, state, o.Beta2)
	buf := state.M.Sqrt().AddScalarInPlace(o.Epsilon)
	suppDiv := state.V.Div(buf)
	if o.Lambda != 0.0 {
		scaledW := weights.ProdScalar(o.Lambda)
		suppDiv.AddInPlace(scaledW)
	}
	weightsNorm := norm(weights)
	adamStepNorm := norm(suppDiv)
	var trustRatio float64 = 1
	if !(weightsNorm == 0.0 || adamStepNorm == 0.0) {
		trustRatio = weightsNorm / adamStepNorm
	}
	state.Buf3.ProdMatrixScalarInPlace(suppDiv, o.Alpha*trustRatio)
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

func norm(grads mat.Matrix) float64 {
	prod := grads.Prod(grads)
	sum := prod.Sum()
	return math.Sqrt(sum.Scalar().F64())
}

func (o *Lamb[T]) OptimizeParams(param *nn.Param) error {
	if param.State == nil {
		param.State = o.newState(param.Value().Shape()...)
	}

	state, ok := param.State.(*State)
	if !ok {
		return fmt.Errorf("unsupported state type: %T, expected %T", param.State, &State{})
	}

	param.SubInPlace(o.calculateParamUpdate(param.Grad(), state, param.Value()))
	param.ZeroGrad()

	return nil
}
