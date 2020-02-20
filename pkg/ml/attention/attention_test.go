// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package attention

import (
	"gonum.org/v1/gonum/floats"
	"math"
	"saientist.dev/spago/pkg/mat"
	"saientist.dev/spago/pkg/ml/ag"
	"testing"
)

func TestSelfAttention(t *testing.T) {
	g := ag.NewGraph()
	qs := []ag.Node{
		g.NewVariable(mat.NewVecDense([]float64{1.1, 0.0, 2.3}), true),
		g.NewVariable(mat.NewVecDense([]float64{2.2, -0.5, 0.3}), true),
		g.NewVariable(mat.NewVecDense([]float64{3.2, 0.5, 0.4}), true),
	}
	ks := []ag.Node{
		g.NewVariable(mat.NewVecDense([]float64{0.0, 1.2, 1.3}), true),
		g.NewVariable(mat.NewVecDense([]float64{4.5, 4.3, 0.2}), true),
		g.NewVariable(mat.NewVecDense([]float64{2.7, 3.6, 2.1}), true),
	}
	vs := []ag.Node{
		g.NewVariable(mat.NewVecDense([]float64{1.2, 2.3, 3.4}), true),
		g.NewVariable(mat.NewVecDense([]float64{2.2, 8.5, 0.0}), true),
		g.NewVariable(mat.NewVecDense([]float64{2.3, 6.5, 3.5}), true),
	}

	attention := SelfAttention(g, qs, ks, vs, math.Sqrt(3))

	if len(attention) != 3 {
		t.Error("The attention doesn't have the expected length")
	}
	if !floats.EqualApprox(attention[0].Value().Data(), []float64{2.22875441063165, 6.68411289826994, 2.82497984315079}, 1.0e-6) {
		t.Error("Attention[0] doesn't match the expected values")
	}
	if !floats.EqualApprox(attention[1].Value().Data(), []float64{2.20637295180029, 8.15650999969648, 0.539678848469417}, 1.0e-6) {
		t.Error("Attention[1] doesn't match the expected values")
	}
	if !floats.EqualApprox(attention[2].Value().Data(), []float64{2.20423303670527, 8.41210390591632, 0.152898186332002}, 1.0e-6) {
		t.Error("Attention[2] doesn't match the expected values")
	}
}
