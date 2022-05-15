// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package adam

import (
	"math"

	"github.com/nlpodyssey/spago/gd"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ gd.MethodConfig = &Config{}

// Config provides configuration settings for an Adam optimizer.
type Config struct {
	gd.MethodConfig
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

var _ gd.Method = &Adam[float32]{}

// Adam implements the Adam gradient descent optimization method.
type Adam[T mat.DType] struct {
	Config
	Alpha    float64
	TimeStep int
	adamw    bool
}

// New returns a new Adam optimizer, initialized according to the given configuration.
func New[T mat.DType](c Config) *Adam[T] {
	adam := &Adam[T]{
		Config: c,
		Alpha:  c.StepSize,
		adamw:  c.Lambda != 0.0,
	}
	adam.IncExample() // initialize 'alpha' coefficient
	return adam
}

// Label returns the enumeration-like value which identifies this gradient descent method.
func (o *Adam[_]) Label() int {
	return gd.Adam
}

const (
	v    int = 0
	m    int = 1
	buf1 int = 2 // contains 'grads.ProdScalar(1.0 - beta1)'
	buf2 int = 3 // contains 'grads.Prod(grads).ProdScalar(1.0 - beta2)'
	buf3 int = 4
)

// NewSupport returns a new support structure with the given dimensions.
func (o *Adam[T]) NewSupport(r, c int) *nn.Payload {
	supp := make([]mat.Matrix, 5)
	supp[v] = mat.NewEmptyDense[T](r, c)
	supp[m] = mat.NewEmptyDense[T](r, c)
	supp[buf1] = mat.NewEmptyDense[T](r, c)
	supp[buf2] = mat.NewEmptyDense[T](r, c)
	supp[buf3] = mat.NewEmptyDense[T](r, c)
	return &nn.Payload{
		Label: o.Label(),
		Data:  supp,
	}
}

// IncExample beats the occurrence of a new example.
func (o *Adam[_]) IncExample() {
	o.TimeStep++
	o.updateAlpha()
}

func (o *Adam[T]) updateAlpha() {
	ts := float64(o.TimeStep)
	o.Alpha = o.StepSize * math.Sqrt(1.0-math.Pow(o.Beta2, ts)) / (1.0 - math.Pow(o.Beta1, ts))
}

// Delta returns the difference between the current params and where the method wants it to be.
func (o *Adam[T]) Delta(param nn.Param) mat.Matrix {
	if o.adamw {
		return o.calcDeltaW(param.Grad(), gd.GetOrSetPayload(param, o).Data, param.Value())
	}
	return o.calcDelta(param.Grad(), gd.GetOrSetPayload(param, o).Data)
}

// v = v*beta1 + grads*(1.0-beta1)
// m = m*beta2 + (grads*grads)*(1.0-beta2)
// d = (v / (sqrt(m) + eps)) * alpha
func (o *Adam[T]) calcDelta(grads mat.Matrix, supp []mat.Matrix) mat.Matrix {
	updateV(grads, supp, o.Beta1)
	updateM(grads, supp, o.Beta2)
	buf := supp[m].Sqrt().AddScalarInPlace(o.Epsilon)
	defer mat.ReleaseMatrix(buf)
	suppDiv := supp[v].Div(buf)
	defer mat.ReleaseMatrix(suppDiv)
	supp[buf3].ProdMatrixScalarInPlace(suppDiv, o.Alpha)
	return supp[buf3]
}

// v = v*beta1 + grads*(1.0-beta1)
// m = m*beta2 + (grads*grads)*(1.0-beta2)
// d = (v / (sqrt(m) + eps))  + (lambda * weights) + alpha
func (o *Adam[T]) calcDeltaW(grads mat.Matrix, supp []mat.Matrix, weights mat.Matrix) mat.Matrix {
	updateV(grads, supp, o.Beta1)
	updateM(grads, supp, o.Beta2)
	buf := supp[m].Sqrt().AddScalarInPlace(o.Epsilon)
	defer mat.ReleaseMatrix(buf)
	suppDiv := supp[v].Div(buf)
	scaledW := weights.ProdScalar(o.Lambda)
	suppDiv.AddInPlace(scaledW)
	defer mat.ReleaseMatrix(suppDiv)
	supp[buf3].ProdMatrixScalarInPlace(suppDiv, o.Alpha)
	return supp[buf3]
}

// v = v*beta1 + grads*(1.0-beta1)
func updateV(grads mat.Matrix, supp []mat.Matrix, beta1 float64) {
	supp[v].ProdScalarInPlace(beta1)
	supp[buf1].ProdMatrixScalarInPlace(grads, 1.0-beta1)
	supp[v].AddInPlace(supp[buf1])
}

// m = m*beta2 + (grads*grads)*(1.0-beta2)
func updateM(grads mat.Matrix, supp []mat.Matrix, beta2 float64) {
	supp[m].ProdScalarInPlace(beta2)
	sqGrad := grads.Prod(grads)
	defer mat.ReleaseMatrix(sqGrad)
	supp[buf2].ProdMatrixScalarInPlace(sqGrad, 1.0-beta2)
	supp[m].AddInPlace(supp[buf2])
}
