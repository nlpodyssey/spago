// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

import (
	"brillion.io/spago/pkg/mat/f64utils"
	"brillion.io/spago/pkg/ml/ag"
	"brillion.io/spago/pkg/ml/nn"
	"brillion.io/spago/pkg/ml/stats"
	"github.com/gosuri/uiprogress"
)

type Evaluator struct {
	model nn.Model
}

func NewEvaluator(model nn.Model) *Evaluator {
	return &Evaluator{
		model: model,
	}
}

// Predict performs the forward pass
func (t *Evaluator) Predict(example Sequence) int {
	g := ag.NewGraph()
	xs := make([]ag.Node, len(example))
	for i, x := range example {
		xs[i] = g.NewScalar(x.Input)
	}
	ys := t.model.NewProc(g).Forward(xs...)
	return f64utils.ArgMax(ys[len(example)-1].Value().Data())
}

func (t *Evaluator) Evaluate(dataset []Sequence) *stats.ClassMetrics {
	uip := uiprogress.New()
	bar := newTestBar(uip, dataset)
	uip.Start()
	defer uip.Stop()
	counter := stats.NewMetricCounter()
	for i := 0; i < len(dataset); i++ {
		sequence := dataset[i]
		if t.Predict(sequence) == sequence[len(sequence)-1].Target {
			counter.IncTruePos()
		} else {
			counter.IncFalsePos()
		}
		bar.Incr()
	}
	return counter
}

func newTestBar(p *uiprogress.Progress, dataset []Sequence) *uiprogress.Bar {
	bar := p.AddBar(len(dataset))
	bar.AppendCompleted().PrependElapsed()
	return bar
}
