// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lamb

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
)

var _ gd.MethodConfig = &Config{}

// Config provides configuration settings for Lamb optimizer.
type Config struct {
	gd.MethodConfig
	StepSize mat.Float
	Beta1    mat.Float
	Beta2    mat.Float
	Epsilon  mat.Float
	Lambda   mat.Float
}

// NewConfig returns a new Lamb Config.
func NewConfig(stepSize, beta1, beta2, epsilon, lambda mat.Float) Config {
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

var _ gd.Method = &Lamb{}

// Lamb implements the Lamb gradient descent optimization method.
type Lamb struct {
	Config
	Alpha    mat.Float
	TimeStep int
}

// New returns a new Lamb optimizer, initialized according to the given configuration.
func New(c Config) *Lamb {
	lamb := &Lamb{
		Config: c,
		Alpha:  c.StepSize,
	}
	lamb.IncExample() // initialize 'alpha' coefficient
	return lamb
}

// Label returns the enumeration-like value which identifies this gradient descent method.
func (o *Lamb) Label() int {
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
func (o *Lamb) NewSupport(r, c int) *nn.Payload {
	supp := make([]mat.Matrix[mat.Float], 5)
	supp[v] = mat.NewEmptyDense[mat.Float](r, c)
	supp[m] = mat.NewEmptyDense[mat.Float](r, c)
	supp[buf1] = mat.NewEmptyDense[mat.Float](r, c)
	supp[buf2] = mat.NewEmptyDense[mat.Float](r, c)
	supp[buf3] = mat.NewEmptyDense[mat.Float](r, c)
	return &nn.Payload{
		Label: o.Label(),
		Data:  supp,
	}
}

// IncExample beats the occurrence of a new example.
func (o *Lamb) IncExample() {
	o.TimeStep++
	o.updateAlpha()
}

func (o *Lamb) updateAlpha() {
	o.Alpha = o.StepSize * mat.Sqrt(1.0-mat.Pow(o.Beta2, mat.Float(o.TimeStep))) / (1.0 - mat.Pow(o.Beta1, mat.Float(o.TimeStep)))
}

// Delta returns the difference between the current params and where the method wants it to be.
func (o *Lamb) Delta(param nn.Param) mat.Matrix[mat.Float] {
	return o.calcDelta(param.Grad(), gd.GetOrSetPayload(param, o).Data, param.Value())
}

// v = v*beta1 + grads*(1.0-beta1)
// m = m*beta2 + (grads*grads)*(1.0-beta2)
// weights = ||params|| / || (v / (sqrt(m) + eps)) + (lambda * weights)
// d = (v / (sqrt(m) + eps)) + (lambda * weights) * alpha
func (o *Lamb) calcDelta(grads mat.Matrix[mat.Float], supp []mat.Matrix[mat.Float], weights mat.Matrix[mat.Float]) mat.Matrix[mat.Float] {
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
	trustRatio := mat.Float(1.0)
	if !(weightsNorm == 0.0 || adamStepNorm == 0.0) {
		trustRatio = weightsNorm / adamStepNorm
	}
	defer mat.ReleaseMatrix(suppDiv)
	supp[buf3].ProdMatrixScalarInPlace(suppDiv, o.Alpha*trustRatio)
	return supp[buf3]
}

// v = v*beta1 + grads*(1.0-beta1)
func updateV(grads mat.Matrix[mat.Float], supp []mat.Matrix[mat.Float], beta1 mat.Float) {
	supp[v].ProdScalarInPlace(beta1)
	supp[buf1].ProdMatrixScalarInPlace(grads, 1.0-beta1)
	supp[v].AddInPlace(supp[buf1])
}

// m = m*beta2 + (grads*grads)*(1.0-beta2)
func updateM(grads mat.Matrix[mat.Float], supp []mat.Matrix[mat.Float], beta2 mat.Float) {
	supp[m].ProdScalarInPlace(beta2)
	sqGrad := grads.Prod(grads)
	defer mat.ReleaseMatrix(sqGrad)
	supp[buf2].ProdMatrixScalarInPlace(sqGrad, 1.0-beta2)
	supp[m].AddInPlace(supp[buf2])
}

func norm(grads mat.Matrix[mat.Float]) mat.Float {
	sum := mat.Float(0.0)
	for _, d := range grads.Data() {
		sum += d * d
	}
	return mat.Sqrt(sum)
}
