// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package evolvingembeddings

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"gonum.org/v1/gonum/floats"
	"testing"
)

// TODO: split test in multiple functions; increase tests coverage
func TestModel_NewAggregateDropAll(t *testing.T) {
	model := New(Config{
		Size:             12,
		PoolingOperation: Min,
		DBPath:           "/tmp/evolvingembeddings_test",
		ForceNewDB:       true,
	})
	wordInContext1 := &WordVectorPair{
		Word:   "foo",
		Vector: mat.NewVecDense([]float64{0.1, 0.2, 0.3, 0.4, 0.5, -0.6, -0.5, 0.8, -0.8, -3, -0.3, -0.4}),
	}
	sameWordInContext2 := &WordVectorPair{
		Word:   "foo",
		Vector: mat.NewVecDense([]float64{0.2, 0.7, 0.5, 0.0, 0.4, 0.5, -0.8, 0.7, -0.3, 0.2, -0.0, -0.9}),
	}

	g := ag.NewGraph()
	proc := model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).(*Processor)
	res := proc.Encode([]string{"foo"})[0]
	if !floats.EqualApprox(res.Value().Data(), wordInContext1.Vector.ZerosLike().Data(), 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}

	model.Aggregate([]*WordVectorPair{wordInContext1})

	g = ag.NewGraph()
	proc = model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).(*Processor)
	res = proc.Encode([]string{"foo"})[0]
	if !floats.EqualApprox(res.Value().Data(), wordInContext1.Vector.Data(), 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}

	model.Aggregate([]*WordVectorPair{sameWordInContext2})

	g = ag.NewGraph()
	proc = model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).(*Processor)
	res = proc.Encode([]string{"foo"})[0]
	if !floats.EqualApprox(res.Value().Data(), []float64{
		0.1, 0.2, 0.3, 0.0, 0.4, -0.6, -0.8, 0.7, -0.8, -3, -0.3, -0.9,
	}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}

	err := model.DropAll()
	if err != nil {
		t.Error(err)
	}

	g = ag.NewGraph()
	proc = model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).(*Processor)
	res = proc.Encode([]string{"foo"})[0]
	if !floats.EqualApprox(res.Value().Data(), []float64{
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	}, 1.0e-6) {
		t.Error("The result doesn't match the expected values")
	}
}
