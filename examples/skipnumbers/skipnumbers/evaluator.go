// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package skipnumbers

import (
	"github.com/gosuri/uiprogress"
	"saientist.dev/spago/pkg/mat"
	"saientist.dev/spago/pkg/mat/f64utils"
	"saientist.dev/spago/pkg/ml/ag"
	"saientist.dev/spago/pkg/ml/stats"
)

type Evaluator struct {
	model *Model
}

func NewEvaluator(model *Model) *Evaluator {
	return &Evaluator{
		model: model,
	}
}

// Predict performs the forward pass
func (t *Evaluator) Predict(example example) int {
	g := ag.NewGraph()
	xs := make([]ag.Node, len(example.xs))
	for i, x := range example.xs {
		xs[i] = g.NewVariable(mat.OneHotVecDense(10, x), false)
	}
	y := t.model.NewProc(g).Forward(xs...)[0]
	return f64utils.ArgMax(y.Value().Data())
}

func (t *Evaluator) Evaluate(dataset []example) *stats.ClassMetrics {
	uip := uiprogress.New()
	bar := newTestBar(uip, dataset)
	uip.Start()
	defer uip.Stop()
	counter := stats.NewMetricCounter()
	for i := 0; i < len(dataset); i++ {
		sequence := dataset[i]
		if t.Predict(sequence) == sequence.y {
			counter.IncTruePos()
		} else {
			counter.IncFalsePos()
		}
		bar.Incr()
	}
	return counter
}

func newTestBar(p *uiprogress.Progress, dataset []example) *uiprogress.Bar {
	bar := p.AddBar(len(dataset))
	bar.AppendCompleted().PrependElapsed()
	return bar
}
