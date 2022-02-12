// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package radam

import (
	"github.com/nlpodyssey/spago/gd"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ gd.MethodConfig = &Config[float32]{}

// Config provides configuration settings for a RAdam optimizer.
type Config[T mat.DType] struct {
	gd.MethodConfig
	StepSize T
	Beta1    T
	Beta2    T
	Epsilon  T
}

// NewConfig returns a new RAdam Config.
// It panics if beta1 or beta2 are not in the range [0.0, 1.0).
func NewConfig[T mat.DType](stepSize, beta1, beta2, epsilon T) Config[T] {
	if !(beta1 >= 0.0 && beta1 < 1.0) {
		panic("adam: `beta1` must be in the range [0.0, 1.0)")
	}
	if !(beta2 >= 0.0 && beta2 < 1.0) {
		panic("adam: `beta2` must be in the range [0.0, 1.0)")
	}
	return Config[T]{
		StepSize: stepSize,
		Beta1:    beta1,
		Beta2:    beta2,
		Epsilon:  epsilon,
	}
}

// NewDefaultConfig returns a new Config with generically reasonable default values.
func NewDefaultConfig[T mat.DType]() Config[T] {
	return Config[T]{
		StepSize: 0.001,
		Beta1:    0.9,
		Beta2:    0.999,
		Epsilon:  1.0e-8,
	}
}

var _ gd.Method[float32] = &RAdam[float32]{}

// RAdam implements the RAdam gradient descent optimization method.
type RAdam[T mat.DType] struct {
	Config[T]
	RoMax    T // The maximum length of the approximated SMA.
	TimeStep int
}

// New returns a new RAdam optimizer, initialized according to the given configuration.
func New[T mat.DType](c Config[T]) *RAdam[T] {
	adam := &RAdam[T]{
		Config:   c,
		RoMax:    2.0/(1.0-c.Beta2) - 1.0,
		TimeStep: 1.0,
	}
	return adam
}

// Label returns the enumeration-like value which identifies this gradient descent method.
func (o *RAdam[_]) Label() int {
	return gd.RAdam
}

const (
	m    int = 0
	v    int = 1
	buf1 int = 2
	buf2 int = 3
	buf3 int = 4
)

// NewSupport returns a new support structure with the given dimensions.
func (o *RAdam[T]) NewSupport(r, c int) *nn.Payload[T] {
	supp := make([]mat.Matrix[T], 5)
	supp[m] = mat.NewEmptyDense[T](r, c)
	supp[v] = mat.NewEmptyDense[T](r, c)
	supp[buf1] = mat.NewEmptyDense[T](r, c)
	supp[buf2] = mat.NewEmptyDense[T](r, c)
	supp[buf3] = mat.NewEmptyDense[T](r, c)
	return &nn.Payload[T]{
		Label: o.Label(),
		Data:  supp,
	}
}

// IncBatch beats the occurrence of a new batch.
func (o *RAdam[_]) IncBatch() {
	o.TimeStep++
}

// Delta returns the difference between the current params and where the method wants it to be.
func (o *RAdam[T]) Delta(param nn.Param[T]) mat.Matrix[T] {
	return o.calcDelta(param.Grad(), gd.GetOrSetPayload[T](param, o).Data)
}

func (o *RAdam[T]) calcDelta(grads mat.Matrix[T], supp []mat.Matrix[T]) mat.Matrix[T] {
	updateM(grads, supp, o.Beta1)
	updateV(grads, supp, o.Beta2)
	sqrtB2T := mat.Sqrt(1.0 - mat.Pow(o.Beta2, T(o.TimeStep)))
	alpha := o.calcAlpha()
	buf := supp[v].Sqrt().AddScalarInPlace(o.Epsilon * sqrtB2T)
	defer mat.ReleaseMatrix(buf)
	suppDiv := supp[m].Div(buf)
	defer mat.ReleaseMatrix(suppDiv)
	supp[buf3].ProdMatrixScalarInPlace(suppDiv, alpha)
	return supp[buf3]
}

// m = m*beta1 + grads*(1.0-beta1)
func updateM[T mat.DType](grads mat.Matrix[T], supp []mat.Matrix[T], beta1 T) {
	supp[m].ProdScalarInPlace(beta1)
	supp[buf1].ProdMatrixScalarInPlace(grads, 1.0-beta1)
	supp[m].AddInPlace(supp[buf1])
}

// v = v*beta2 + (grads*grads)*(1.0-beta2)
func updateV[T mat.DType](grads mat.Matrix[T], supp []mat.Matrix[T], beta2 T) {
	supp[v].ProdScalarInPlace(beta2)
	sqGrad := grads.Prod(grads)
	defer mat.ReleaseMatrix(sqGrad)
	supp[buf2].ProdMatrixScalarInPlace(sqGrad, 1.0-beta2)
	supp[v].AddInPlace(supp[buf2])
}

func (o *RAdam[T]) calcAlpha() T {
	timeStep := T(o.TimeStep)
	b1T := mat.Pow(o.Beta1, timeStep)
	b2T := mat.Pow(o.Beta2, timeStep)
	ro := o.RoMax - 2.0*timeStep*b2T/(1.0-b2T)
	var rect T = 1.0
	if ro > 4.0 { // i.e. if the variance is tractable
		rect = mat.Sqrt((ro - 4.0) * (ro - 2.0) * o.RoMax / ((o.RoMax - 4.0) * (o.RoMax - 2.0) * ro))
	}
	return o.StepSize * rect * mat.Sqrt(1.0-b2T) / (1.0 - b1T)
}
