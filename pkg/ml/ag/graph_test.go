// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"testing"
)

func TestNewGraph(t *testing.T) {
	g := NewGraph()
	if !(g.maxID == -1) {
		t.Errorf("maxID doesn't match the expected value.")
	}
	if !(g.curTimeStep == 0) {
		t.Errorf("curTimeStep doesn't match the expected value.")
	}
	if !(g.nodes == nil) {
		t.Errorf("nodes doesn't match the expected value.")
	}
}

func TestGraph_TimeStep(t *testing.T) {
	g := NewGraph()
	if !(g.TimeStep() == 0) {
		t.Errorf("The graph time-step doesn't match the expected value.")
	}
	a := g.NewVariable(mat.NewVecDense([]float64{1.0}), false)
	if !(a.getTimeStep() == 0) {
		t.Errorf("The node time-step doesn't match the expected value.")
	}
	g.IncTimeStep()
	if !(g.TimeStep() == 1) {
		t.Errorf("The graph time-step doesn't match the expected value.")
	}
	b := g.NewVariable(mat.NewVecDense([]float64{2.0}), false)
	if !(b.getTimeStep() == 1) {
		t.Errorf("The node time-step doesn't match the expected value.")
	}
	g.IncTimeStep()
	if !(g.TimeStep() == 2) {
		t.Errorf("The graph time-step doesn't match the expected value.")
	}
	c := g.NewVariable(mat.NewVecDense([]float64{3.0}), false)
	if !(c.getTimeStep() == 2) {
		t.Errorf("The node time-step doesn't match the expected value.")
	}
	g.IncTimeStep()
	if !(g.TimeStep() == 3) {
		t.Errorf("The graph time-step doesn't match the expected value.")
	}
	d := g.NewVariable(mat.NewVecDense([]float64{4.0}), false)
	if !(d.getTimeStep() == 3) {
		t.Errorf("The node time-step doesn't match the expected value.")
	}
}
