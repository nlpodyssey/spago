// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package adam

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
	"math"
)

var _ gd.MethodConfig = &Config{}

type Config struct {
	gd.MethodConfig
	StepSize float64
	Beta1    float64
	Beta2    float64
	Epsilon  float64
}

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

func NewDefaultConfig() Config {
	return Config{
		StepSize: 0.001,
		Beta1:    0.9,
		Beta2:    0.999,
		Epsilon:  1.0e-8,
	}
}

var _ gd.Method = &Adam{}

type Adam struct {
	Config
	Alpha float64
	Count int
}

func New(c Config) *Adam {
	adam := &Adam{
		Config: c,
		Alpha:  c.StepSize,
	}
	adam.IncExample() // initialize 'alpha' coefficient
	return adam
}

func (o *Adam) Label() int {
	return gd.Adam
}

const (
	v    int = 0
	m    int = 1
	buf1 int = 2 // contains 'grads.ProdScalar(1.0 - beta1)'
	buf2 int = 3 // contains 'grads.Prod(grads).ProdScalar(1.0 - beta2)'
	buf3 int = 4
)

func (o *Adam) NewSupport(r, c int) *nn.Payload {
	supp := make([]mat.Matrix, 5)
	supp[v] = mat.NewEmptyDense(r, c)
	supp[m] = mat.NewEmptyDense(r, c)
	supp[buf1] = mat.NewEmptyDense(r, c)
	supp[buf2] = mat.NewEmptyDense(r, c)
	supp[buf3] = mat.NewEmptyDense(r, c)
	return &nn.Payload{
		Label: o.Label(),
		Data:  supp,
	}
}

func (o *Adam) IncExample() {
	o.Count++
	o.updateAlpha()
}

func (o *Adam) updateAlpha() {
	o.Alpha = o.StepSize * math.Sqrt(1.0-math.Pow(o.Beta2, float64(o.Count))) / (1.0 - math.Pow(o.Beta1, float64(o.Count)))
}

func (o *Adam) Delta(param *nn.Param) mat.Matrix {
	return o.calcDelta(param.Grad(), gd.GetOrSetPayload(param, o).Data)
}

// v = v*beta1 + grads*(1.0-beta1)
// m = m*beta2 + (grads*grads)*(1.0-beta2)
// d = (v / (sqrt(m) + eps)) * alpha
func (o *Adam) calcDelta(grads mat.Matrix, supp []mat.Matrix) mat.Matrix {
	updateV(grads, supp, o.Beta1)
	updateM(grads, supp, o.Beta2)
	buf := supp[m].Sqrt().AddScalarInPlace(o.Epsilon)
	defer mat.ReleaseDense(buf.(*mat.Dense))
	suppDiv := supp[v].Div(buf)
	defer mat.ReleaseDense(suppDiv.(*mat.Dense))
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
	defer mat.ReleaseDense(sqGrad.(*mat.Dense))
	supp[buf2].ProdMatrixScalarInPlace(sqGrad, 1.0-beta2)
	supp[m].AddInPlace(supp[buf2])
}
