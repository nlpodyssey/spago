// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package optimizer

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/optimizer/clipper"
)

type Optimizer struct {
	model       nn.Model
	strategy    Strategy // optimization method, e.g. Momentum, Adam, RAdam, or SGD
	gradClipper clipper.GradClipper
}

func New(model nn.Model, opts ...func(*Optimizer)) *Optimizer {
	optimizer := &Optimizer{
		model: model,
	}
	for _, option := range opts {
		option(optimizer)
	}
	return optimizer
}

func WithMethod(method Strategy) func(*Optimizer) {
	return func(o *Optimizer) {
		o.strategy = method
	}
}

func (o *Optimizer) WithClipGradByValue(value float64) *Optimizer {
	o.gradClipper = &clipper.ClipValue{Value: value}
	return o
}

func (o *Optimizer) WithClipGradByNorm(max, normType float64) *Optimizer {
	o.gradClipper = &clipper.ClipNorm{
		MaxNorm:  max,
		NormType: normType,
	}
	return o
}

// Optimize performs the optimization step.
func (o *Optimizer) Optimize() error {
	if o.strategy == nil {
		return fmt.Errorf("optimizer: strategy not set")
	}

	visited := make(map[*nn.Param]struct{})
	var params []*nn.Param

	nn.ForEachParam(o.model, func(param *nn.Param) {
		if !param.HasGrad() {
			return
		}
		if _, ok := visited[param]; !ok {
			params = append(params, param)
			visited[param] = struct{}{}
		}
	})

	o.clipGradsInPlace(params)
	o.updateParams(params)
	return nil
}

func (o *Optimizer) updateParams(params []*nn.Param) {
	var wg sync.WaitGroup
	ch := make(chan *nn.Param, runtime.NumCPU())

	wg.Add(len(params))
	go func() {
		for param := range ch {
			param.ApplyDelta(o.strategy.CalcDelta(param))
			param.ZeroGrad()
			wg.Done()
		}
	}()

	for _, param := range params {
		ch <- param
	}

	close(ch)
	wg.Wait()
}

func (o *Optimizer) clipGradsInPlace(params []*nn.Param) {
	if o.gradClipper == nil {
		return
	}

	var gs []mat.Matrix
	for _, param := range params {
		gs = append(gs, param.Grad())
	}

	o.gradClipper.Clip(gs)
}

func (o *Optimizer) IncExample() {
	if method, ok := o.strategy.(interface{ IncExample() }); ok {
		method.IncExample()
	}
}

func (o *Optimizer) IncBatch() {
	if method, ok := o.strategy.(interface{ IncBatch() }); ok {
		method.IncBatch()
	}
}

func (o *Optimizer) IncEpoch() {
	if method, ok := o.strategy.(interface{ IncEpoch() }); ok {
		method.IncEpoch()
	}
}
