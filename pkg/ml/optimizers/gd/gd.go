// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gd

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd/clipper"
	"sync"
)

// GradientDescent implements Gradients Descent (GD) optimization.
type GradientDescent struct {
	method           Method // optimization method (SGD, AdaGrad, Adam, ...)
	gradClipper      clipper.GradClipper
	paramsIterator   nn.ParamsIterator
	paramsToOptimize []*nn.Param
}

type Option func(*GradientDescent)

func ClipGradByValue(value float64) Option {
	return func(f *GradientDescent) {
		f.gradClipper = &clipper.ClipValue{Value: value}
	}
}

func ClipGradByNorm(max, normType float64) Option {
	return func(f *GradientDescent) {
		f.gradClipper = &clipper.ClipNorm{
			MaxNorm:  max,
			NormType: normType,
		}
	}
}

// NewOptimizer returns a new GradientDescent optimizer. The gradient clipper can be set to nil.
func NewOptimizer(method Method, paramsIterator nn.ParamsIterator, opts ...Option) *GradientDescent {
	optimizer := &GradientDescent{
		method:           method,
		paramsIterator:   paramsIterator,
		paramsToOptimize: make([]*nn.Param, 0),
	}
	for _, opt := range opts {
		opt(optimizer)
	}
	return optimizer
}

// Optimize optimize the params, applying the optional gradient clipping.
// After the optimization the params have zero gradients.
func (o *GradientDescent) Optimize() {
	o.paramsToOptimize = o.paramsIterator.ParamsList()
	if o.paramsToOptimize == nil {
		return
	}
	o.clipGrads()
	o.updateParams()
	o.paramsToOptimize = nil
}

// updateParamsSerial applies the optimization method to all the observed parameters.
func (o *GradientDescent) updateParamsSerial() {
	for _, param := range o.paramsToOptimize {
		if param.HasGrad() {
			delta := o.method.Delta(param) // important: don't release delta here
			param.ApplyDelta(delta)
			param.ZeroGrad()
		}
	}
}

// updateParams applies the optimization method to all the observed parameters concurrently.
// TODO: distribute the workload proportionately to the number of available CPUs?
func (o *GradientDescent) updateParams() {
	var wg sync.WaitGroup
	for _, param := range o.paramsToOptimize {
		if param.HasGrad() {
			wg.Add(1)
			go func(param *nn.Param) {
				defer wg.Done()
				delta := o.method.Delta(param)
				param.ApplyDelta(delta)
				param.ZeroGrad()
			}(param)
		}
	}
	wg.Wait()
}

// clipGrad applies the gradient clipping to all the observed parameters.
func (o *GradientDescent) clipGrads() {
	if o.gradClipper == nil {
		return
	}
	var gs []mat.Matrix
	for _, param := range o.paramsToOptimize {
		if param.HasGrad() { // don't consider grad at zero
			gs = append(gs, param.Grad())
		}
	}
	o.gradClipper.Clip(gs)
}

// IncExample beats the occurrence of a new example.
func (o *GradientDescent) IncExample() {
	if method, ok := o.method.(ExampleScheduler); ok {
		method.IncExample()
	}
}

// IncBatch beats the occurrence of a new batch.
func (o *GradientDescent) IncBatch() {
	if method, ok := o.method.(BatchScheduler); ok {
		method.IncBatch()
	}
}

// IncEpoch beats the occurrence of a new epoch.
func (o *GradientDescent) IncEpoch() {
	if method, ok := o.method.(EpochScheduler); ok {
		method.IncEpoch()
	}
}
