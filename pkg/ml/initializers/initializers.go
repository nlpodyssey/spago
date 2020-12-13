// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package initializers

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/mat/rand/normal"
	"github.com/nlpodyssey/spago/pkg/mat/rand/uniform"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"math"
)

// Gain returns a coefficient that help to initialize the params in a way to keep gradients stable.
// Use it to find the gain value for Xavier initializations.
func Gain(f ag.OpName) float64 {
	switch f {
	case ag.OpSigmoid:
		return 1.0
	case ag.OpReLU:
		return math.Sqrt(2.0)
	case ag.OpTanh:
		return 5.0 / 3
	default:
		return 1.0
	}
}

// Uniform fills the input matrix m with a uniform distribution where a is the lower bound and b is the upper bound.
func Uniform(m mat.Matrix, min, max float64, generator *rand.LockedRand) {
	dist := uniform.New(min, max, generator)
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Columns(); j++ {
			m.Set(i, j, dist.Next())
		}
	}
}

func Normal(m mat.Matrix, mean, std float64, generator *rand.LockedRand) {
	dist := normal.New(std, mean, generator)
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Columns(); j++ {
			m.Set(i, j, dist.Next())
		}
	}
}

// Constant fills the input matrix with the value n.
func Constant(m mat.Matrix, n float64) {
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Columns(); j++ {
			m.Set(i, j, n)
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

// XavierUniform fills the input `m` with values according to the method described in `Understanding the difficulty of training deep
// feedforward  neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform distribution.
func XavierUniform(m mat.Matrix, gain float64, generator *rand.LockedRand) {
	rows, cols := m.Dims()
	a := gain * math.Sqrt(6.0/float64(rows+cols))
	dist := uniform.New(-a, a, generator)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.Set(i, j, dist.Next())
		}
	}
}

func XavierNormal(m mat.Matrix, gain float64, generator *rand.LockedRand) {
	rows, cols := m.Dims()
	std := gain * math.Sqrt(2.0/float64(rows+cols))
	dist := normal.New(std, 0, generator)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.Set(i, j, dist.Next())
		}
	}
}

func Achlioptas(m mat.Matrix, generator *rand.LockedRand) {
	dist := uniform.New(0.0, 1.0, generator)
	lower := 1.0 / 6.0
	upper := 1.0 - lower
	a := math.Sqrt(3.0)
	rows, cols := m.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			r := dist.Next()
			if r < lower {
				m.Set(i, j, -a)
			} else if r > upper {
				m.Set(i, j, a)
			} else {
				m.Set(i, j, 0.0)
			}
		}
	}
}
