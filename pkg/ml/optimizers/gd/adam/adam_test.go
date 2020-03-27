// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package adam

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func Test_IncExample(t *testing.T) {
	updater := New(NewConfig(
		0.001,  // step size
		0.9,    // beta1
		0.999,  // beta2
		1.0e-8, // epsilon
	))

	if !floats.EqualWithinAbsOrRel(updater.Alpha, 3.1623e-4, 1.0e-08, 1.0e-08) {
		t.Error("The Alpha doesn't match the expected value")
	}
}

func Test_Update(t *testing.T) {
	updater := New(NewConfig(
		0.001,  // step size
		0.9,    // beta1
		0.999,  // beta2
		1.0e-8, // epsilon
	))

	params := mat.NewVecDense([]float64{0.4, 0.4, 0.5, 1.0, 0.8})
	grads := mat.NewVecDense([]float64{0.9, 0.7, 0.4, 0.8, 0.1})

	supp := updater.NewSupport(params.Dims()).Data
	supp[v].SetData([]float64{0.7, 0.8, 0.5, 0.3, 0.2})
	supp[m].SetData([]float64{1.0, 0.4, 0.7, 0.0, 0.2})

	params.SubInPlace(updater.calcDelta(grads, supp))

	if !floats.EqualApprox(params.Data(), []float64{0.399772, 0.399605, 0.4998147, 0.995625, 0.799865}, 1.0e-6) {
		t.Error("The updated params don't match the expected values")
	}
}

func Test_Update2(t *testing.T) {
	updater := New(NewConfig(
		0.001,  // step size
		0.9,    // beta1
		0.999,  // beta2
		1.0e-8, // epsilon
	))

	params := mat.NewDense(3, 3, []float64{
		1.4, 1.3, 0,
		-0.8, 0.16, 0.65,
		0.7, -0.4, 0.2,
	})

	grads := mat.NewDense(3, 3, []float64{
		0.5, 0.3, -0.1,
		-0.6, -0.4, -1.0,
		0.5, -0.6, 0.1,
	})

	supp := updater.NewSupport(params.Dims()).Data

	// === First iteration

	params.SubInPlace(updater.calcDelta(grads, supp))

	if !floats.EqualApprox(supp[v].Data(), []float64{
		0.05, 0.03, -0.01,
		-0.06, -0.04, -0.1,
		0.05, -0.06, 0.01,
	}, 1.0e-6) {
		t.Error("The first order moments don't match the expected values (first iteration)")
	}

	if !floats.EqualApprox(supp[m].Data(), []float64{
		0.00025, 9.0e-05, 1e-05,
		0.00036, 0.00016, 0.001,
		0.00025, 0.00036, 1e-05,
	}, 1.0e-6) {
		t.Error("The second order moments don't match the expected values (first iteration)")
	}

	if !floats.EqualApprox(params.Data(), []float64{
		1.39900000063246, 1.29900000105409, 0.000999996837732,
		-0.799000000527046, 0.160999999209431, 0.650999999683772,
		0.699000000632455, -0.399000000527046, 0.199000003162268,
	}, 1.0e-6) {
		t.Error("The updated params don't match the expected values (first iteration)")
	}

	// === Second iteration

	updater.IncExample()

	grads2 := mat.NewDense(3, 3, []float64{
		0.7, 0.44, -0.66,
		-0.56, 0.4, 1.4,
		0.44, 1.44, 2.44,
	})

	params.SubInPlace(updater.calcDelta(grads2, supp))

	if !floats.EqualApprox(supp[v].Data(), []float64{
		0.115, 0.071, -0.075,
		-0.11, 0.004, 0.05,
		0.089, 0.09, 0.253,
	}, 1.0e-6) {
		t.Error("The first order moments don't match the expected values (second iteration)")
	}

	if !floats.EqualApprox(supp[m].Data(), []float64{
		0.00073975, 0.00028351, 0.00044559,
		0.00067324, 0.00031984, 0.002959,
		0.00044335, 0.00243324, 0.00596359,
	}, 1.0e-6) {
		t.Error("The second order moments don't match the expected values (second iteration)")
	}

	if !floats.EqualApprox(params.Data(), []float64{
		1.39800503520615, 1.29800773828614, 0.001836073567191,
		-0.798002391527689, 0.160947367659913, 0.650783702812203,
		0.698005353030075, -0.399429341903112, 0.198229065305535,
	}, 1.0e-5) {
		t.Error("The updated params don't match the expected values (second iteration)")
	}
}
