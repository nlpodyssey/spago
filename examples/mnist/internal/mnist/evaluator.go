// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mnist

import (
	"brillion.io/spago/pkg/mat/f64utils"
	"brillion.io/spago/pkg/ml/ag"
	"brillion.io/spago/pkg/ml/nn"
	"brillion.io/spago/pkg/ml/stats"
	"brillion.io/spago/third_party/GoMNIST"
	"github.com/gosuri/uiprogress"
)

type Evaluator struct {
	model nn.Model
}

func NewEvaluator(model nn.Model) *Evaluator {
	e := &Evaluator{
		model: model,
	}
	return e
}

// Predict performs the forward pass and returns the predict label
func (t *Evaluator) Predict(image GoMNIST.RawImage) int {
	g := ag.NewGraph()
	x := g.NewVariable(normalize(image), false)
	y := t.model.NewProc(g).Forward(x)[0]
	return f64utils.ArgMax(y.Value().Data())
}

func (t *Evaluator) Evaluate(dataset *GoMNIST.Set) *stats.ClassMetrics {
	uip := uiprogress.New()
	bar := newTestBar(uip, dataset)
	uip.Start()
	defer uip.Stop()

	counter := stats.NewMetricCounter()
	for i := 0; i < dataset.Count(); i++ {
		image, label := dataset.Get(i)
		if t.Predict(image) == int(label) {
			counter.IncTruePos()
		} else {
			counter.IncFalsePos()
		}
		bar.Incr()
	}
	return counter
}

func newTestBar(p *uiprogress.Progress, dataset *GoMNIST.Set) *uiprogress.Bar {
	bar := p.AddBar(dataset.Count())
	bar.AppendCompleted().PrependElapsed()
	return bar
}
