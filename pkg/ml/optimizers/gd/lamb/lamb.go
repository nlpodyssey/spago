// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lamb

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
)

var _ gd.MethodConfig = &Config[float32]{}

// Config provides configuration settings for Lamb optimizer.
type Config[T mat.DType] struct {
	gd.MethodConfig
	StepSize T
	Beta1    T
	Beta2    T
	Epsilon  T
	Lambda   T
}

// NewConfig returns a new Lamb Config.
func NewConfig[T mat.DType](stepSize, beta1, beta2, epsilon, lambda T) Config[T] {
	if !(beta1 >= 0.0 && beta1 < 1.0) {
		panic("lamb: `beta1` must be in the range [0.0, 1.0)")
	}
	if !(beta2 >= 0.0 && beta2 < 1.0) {
		panic("lamb: `beta2` must be in the range [0.0, 1.0)")
	}
	return Config[T]{
		StepSize: stepSize,
		Beta1:    beta1,
		Beta2:    beta2,
		Epsilon:  epsilon,
		Lambda:   lambda,
	}
}

// NewDefaultConfig returns a new Config with generically reasonable default values.
func NewDefaultConfig[T mat.DType]() Config[T] {
	return Config[T]{
		StepSize: 0.001,
		Beta1:    0.9,
		Beta2:    0.999,
		Epsilon:  1.0e-8,
		Lambda:   0.1,
	}
}

var _ gd.Method[float32] = &Lamb[float32]{}

// Lamb implements the Lamb gradient descent optimization method.
type Lamb[T mat.DType] struct {
	Config[T]
	Alpha    T
	TimeStep int
}

// New returns a new Lamb optimizer, initialized according to the given configuration.
func New[T mat.DType](c Config[T]) *Lamb[T] {
	lamb := &Lamb[T]{
		Config: c,
		Alpha:  c.StepSize,
	}
	lamb.IncExample() // initialize 'alpha' coefficient
	return lamb
}

// Label returns the enumeration-like value which identifies this gradient descent method.
func (o *Lamb[_]) Label() int {
	return gd.Lamb
}

const (
	v    int = 0
	m    int = 1
	buf1 int = 2 // contains 'grads.ProdScalar(1.0 - beta1)'
	buf2 int = 3 // contains 'grads.Prod(grads).ProdScalar(1.0 - beta2)'
	buf3 int = 4
)

// NewSupport returns a new support structure with the given dimensions.
func (o *Lamb[T]) NewSupport(r, c int) *nn.Payload[T] {
	supp := make([]mat.Matrix[T], 5)
	supp[v] = mat.NewEmptyDense[T](r, c)
	supp[m] = mat.NewEmptyDense[T](r, c)
	supp[buf1] = mat.NewEmptyDense[T](r, c)
	supp[buf2] = mat.NewEmptyDense[T](r, c)
	supp[buf3] = mat.NewEmptyDense[T](r, c)
	return &nn.Payload[T]{
		Label: o.Label(),
		Data:  supp,
	}
}

// IncExample beats the occurrence of a new example.
func (o *Lamb[_]) IncExample() {
	o.TimeStep++
	o.updateAlpha()
}

func (o *Lamb[T]) updateAlpha() {
	o.Alpha = o.StepSize * mat.Sqrt(1.0-mat.Pow(o.Beta2, T(o.TimeStep))) / (1.0 - mat.Pow(o.Beta1, T(o.TimeStep)))
}

// Delta returns the difference between the current params and where the method wants it to be.
func (o *Lamb[T]) Delta(param nn.Param[T]) mat.Matrix[T] {
	return o.calcDelta(param.Grad(), gd.GetOrSetPayload[T](param, o).Data, param.Value())
}

// v = v*beta1 + grads*(1.0-beta1)
// m = m*beta2 + (grads*grads)*(1.0-beta2)
// weights = ||params|| / || (v / (sqrt(m) + eps)) + (lambda * weights)
// d = (v / (sqrt(m) + eps)) + (lambda * weights) * alpha
func (o *Lamb[T]) calcDelta(grads mat.Matrix[T], supp []mat.Matrix[T], weights mat.Matrix[T]) mat.Matrix[T] {
	updateV(grads, supp, o.Beta1)
	updateM(grads, supp, o.Beta2)
	buf := supp[m].Sqrt().AddScalarInPlace(o.Epsilon)
	defer mat.ReleaseMatrix(buf)
	suppDiv := supp[v].Div(buf)
	if o.Lambda != 0.0 {
		scaledW := weights.ProdScalar(o.Lambda)
		suppDiv.AddInPlace(scaledW)
	}
	weightsNorm := norm(weights)
	adamStepNorm := norm(suppDiv)
	trustRatio := T(1.0)
	if !(weightsNorm == 0.0 || adamStepNorm == 0.0) {
		trustRatio = weightsNorm / adamStepNorm
	}
	defer mat.ReleaseMatrix(suppDiv)
	supp[buf3].ProdMatrixScalarInPlace(suppDiv, o.Alpha*trustRatio)
	return supp[buf3]
}

// v = v*beta1 + grads*(1.0-beta1)
func updateV[T mat.DType](grads mat.Matrix[T], supp []mat.Matrix[T], beta1 T) {
	supp[v].ProdScalarInPlace(beta1)
	supp[buf1].ProdMatrixScalarInPlace(grads, 1.0-beta1)
	supp[v].AddInPlace(supp[buf1])
}

// m = m*beta2 + (grads*grads)*(1.0-beta2)
func updateM[T mat.DType](grads mat.Matrix[T], supp []mat.Matrix[T], beta2 T) {
	supp[m].ProdScalarInPlace(beta2)
	sqGrad := grads.Prod(grads)
	defer mat.ReleaseMatrix(sqGrad)
	supp[buf2].ProdMatrixScalarInPlace(sqGrad, 1.0-beta2)
	supp[m].AddInPlace(supp[buf2])
}

func norm[T mat.DType](grads mat.Matrix[T]) T {
	sum := T(0.0)
	for _, d := range grads.Data() {
		sum += d * d
	}
	return mat.Sqrt(sum)
}
