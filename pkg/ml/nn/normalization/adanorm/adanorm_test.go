// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package adanorm

import (
	"fmt"
	"saientist.dev/spago/pkg/mat"
	"saientist.dev/spago/pkg/ml/ag"
	"testing"
)

func TestModel_Forward(t *testing.T) {

	model := New(0.8)
	g := ag.NewGraph()

	// == Forward

	x1 := g.NewVariable(mat.NewVecDense([]float64{1.0, 2.0, 0.0, 4.0, 4.0, 1.0}), true)
	x2 := g.NewVariable(mat.NewVecDense([]float64{3.0, 2.0, 1.0, 6.0, 2.0, 4.0}), true)
	x3 := g.NewVariable(mat.NewVecDense([]float64{6.0, 2.0, 5.0, 1.0, 3.0, 1.0}), true)

	y := model.NewProc(g).Forward(x1, x2, x3)
	_ = y
	for _, yi := range y {
		fmt.Println(yi.Value())
	}

	// TODO: write tests
}
