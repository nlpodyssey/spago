// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package initializers

import (
	"github.com/saientist/spago/pkg/mat"
	"github.com/saientist/spago/pkg/mat/rnd/normal"
	"github.com/saientist/spago/pkg/mat/rnd/uniform"
	"github.com/saientist/spago/pkg/ml/ag"
	"golang.org/x/exp/rand"
	"math"
)

// Gain returns a coefficient that help to initialize the params in a way to keep gradients stable.
// Use it to find the gain value for Xavier initializations.
func Gain(f ag.OpName) float64 {
	switch f {
	case ag.Sigmoid:
		return 1.0
	case ag.ReLU:
		return math.Sqrt(2.0)
	case ag.Tanh:
		return 5.0 / 3
	default:
		return 1.0
	}
}

// Uniform fills the input matrix m with a uniform distribution where a is the lower bound and b is the upper bound.
func Uniform(m mat.Matrix, min, max float64, source rand.Source) {
	dist := uniform.New(min, max, source)
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Columns(); j++ {
			m.Set(dist.Next(), i, j)
		}
	}
}

// Uniform fills the input matrix m with a uniform distribution where a is the lower bound and b is the upper bound.
func Normal(m mat.Matrix, mean, std float64, source rand.Source) {
	dist := normal.New(0, std, source)
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Columns(); j++ {
			m.Set(dist.Next(), i, j)
		}
	}
}

// Constant fills the input matrix with the value n.
func Constant(m mat.Matrix, n float64) {
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Columns(); j++ {
			m.Set(n, i, j)
		}
	}
}

// Ones fills the input matrix with the scalar value `1`.
func Ones(m mat.Matrix) {
	Constant(m, 1.0)
}

// Zeros fills the input matrix with the scalar value `0`.
func Zeros(m mat.Matrix) {
	m.Zeros()
}

// Fills the input `m` with values according to the method described in `Understanding the difficulty of training deep
// feedforward  neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform distribution.
func XavierUniform(m mat.Matrix, gain float64, source rand.Source) {
	a := gain * math.Sqrt(6.0/float64(m.Rows()+m.Columns()))
	dist := uniform.New(-a, a, source)
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Columns(); j++ {
			m.Set(dist.Next(), i, j)
		}
	}
}

func XavierNormal(m mat.Matrix, gain float64, source rand.Source) {
	std := gain * math.Sqrt(2.0/float64(m.Rows()+m.Columns()))
	dist := normal.New(std, 0, source)
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Columns(); j++ {
			m.Set(dist.Next(), i, j)
		}
	}
}

func Achlioptas(m mat.Matrix, source rand.Source) {
	dist := uniform.New(0.0, 1.0, source)
	lower := 1.0 / 6.0
	upper := 1.0 - lower
	a := math.Sqrt(3.0)
	rows, cols := m.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			r := dist.Next()
			if r < lower {
				m.Set(-a, i, j)
			} else if r > upper {
				m.Set(a, i, j)
			} else {
				m.Set(0.0, i, j)
			}
		}
	}
}
