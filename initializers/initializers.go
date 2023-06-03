// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package initializers

import (
	"math"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/mat/rand"
	"github.com/nlpodyssey/spago/mat/rand/normal"
	"github.com/nlpodyssey/spago/mat/rand/uniform"
	"github.com/nlpodyssey/spago/nn/activation"
)

var sqrt2 = math.Sqrt(2.0)

// Gain returns a coefficient that help to initialize the params in a way to keep gradients stable.
// Use it to find the gain value for Xavier initializations.
func Gain(f activation.Activation) float64 {
	switch f {
	case activation.Sigmoid:
		return 1.0
	case activation.ReLU:
		return sqrt2
	case activation.Tanh:
		return 5.0 / 3
	default:
		return 1.0
	}
}

// Uniform fills the input matrix m with a uniform distribution where a is the lower bound and b is the upper bound.
//
// The matrix is returned for convenience.
func Uniform(m mat.Matrix, min, max float64, generator *rand.LockedRand) mat.Matrix {
	dist := uniform.New(min, max, generator)
	for i := 0; i < m.Shape()[0]; i++ {
		for j := 0; j < m.Shape()[1]; j++ {
			m.SetScalar(float.Interface(dist.Next()), i, j)
		}
	}
	return m
}

// Normal fills the input matrix with random samples from a normal (Gaussian)
// distribution.
//
// The matrix is returned for convenience.
func Normal(m mat.Matrix, mean, std float64, generator *rand.LockedRand) mat.Matrix {
	dist := normal.New(std, mean, generator)
	for i := 0; i < m.Shape()[0]; i++ {
		for j := 0; j < m.Shape()[1]; j++ {
			m.SetScalar(float.Interface(dist.Next()), i, j)
		}
	}
	return m
}

// Constant fills the input matrix with the value n.
//
// The matrix is returned for convenience.
func Constant(m mat.Matrix, n float64) mat.Matrix {
	c := m.NewScalar(n)
	for i := 0; i < m.Shape()[0]; i++ {
		for j := 0; j < m.Shape()[1]; j++ {
			m.SetAt(c, i, j)
		}
	}
	return m
}

// Ones fills the input matrix with the scalar value `1`.
//
// The matrix is returned for convenience.
func Ones(m mat.Matrix) mat.Matrix {
	return Constant(m, 1)
}

// Zeros fills the input matrix with the scalar value `0`.
//
// The matrix is returned for convenience.
func Zeros(m mat.Matrix) mat.Matrix {
	m.Zeros()
	return m
}

// XavierUniform fills the input `m` with values according to the method described in `Understanding the difficulty of training deep
// feedforward  neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform distribution.
//
// The matrix is returned for convenience.
func XavierUniform(m mat.Matrix, gain float64, generator *rand.LockedRand) mat.Matrix {
	shape := m.Shape()
	rows, cols := shape[0], shape[1]
	a := gain * math.Sqrt(6.0/float64(rows+cols))
	dist := uniform.New(-a, a, generator)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.SetScalar(float.Interface(dist.Next()), i, j)
		}
	}
	return m
}

// XavierNormal fills the input matrix with values according to the method
// described in "Understanding the difficulty of training deep feedforward
// neural networks" - Glorot, X. & Bengio, Y. (2010), using a normal
// distribution.
//
// The matrix is returned for convenience.
func XavierNormal(m mat.Matrix, gain float64, generator *rand.LockedRand) mat.Matrix {
	shape := m.Shape()
	rows, cols := shape[0], shape[1]
	std := gain * math.Sqrt(2.0/float64(rows+cols))
	dist := normal.New(std, 0, generator)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.SetScalar(float.Interface(dist.Next()), i, j)
		}
	}
	return m
}

// Achlioptas fills the input matrix with values according to the mthod
// described on "Database-friendly random projections: Johnson-Lindenstrauss
// with binary coins", by Dimitris Achlioptas 2001
// (https://core.ac.uk/download/pdf/82724427.pdf)
//
// The matrix is returned for convenience.
func Achlioptas(m mat.Matrix, generator *rand.LockedRand) mat.Matrix {
	dist := uniform.New(0.0, 1.0, generator)
	lower := 1.0 / 6.0
	upper := 1.0 - lower

	sqrt3 := math.Sqrt(3)
	a := float.Interface(sqrt3)
	negA := float.Interface(-sqrt3)
	zero := float.Interface(0.0)

	shape := m.Shape()
	rows, cols := shape[0], shape[1]
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			r := dist.Next()
			if r < lower {
				m.SetScalar(negA, i, j)
			} else if r > upper {
				m.SetScalar(a, i, j)
			} else {
				m.SetScalar(zero, i, j)
			}
		}
	}

	return m
}
