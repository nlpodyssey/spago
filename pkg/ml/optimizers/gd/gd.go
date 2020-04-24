// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gd

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd/clipper"
	"sync"
)

// Gradients Descent (GD) Optimizer
type GradientDescent struct {
	// the optimization method (SGD, AdaGrad, Adam, ...)
	method OptimizationMethod
	// gradient clipper
	gradClipper clipper.GradClipper
	// set of observed optimizable parameters
	observed map[Optimizable]bool
}

// NewOptimizer returns a new GradientDescent optimizer. The gradient clipper can be set to nil.
func NewOptimizer(method OptimizationMethod, gradClipper clipper.GradClipper) *GradientDescent {
	return &GradientDescent{
		method:      method,
		gradClipper: gradClipper,
		observed:    make(map[Optimizable]bool),
	}
}

// Track tracks the parameters to optimize.
func (o *GradientDescent) Track(vs ...Optimizable) {
	for _, v := range vs {
		if v.RequiresGrad() {
			o.observed[v] = true
		}
	}
}

// Untrack avoid the given parameters to be optimized.
func (o *GradientDescent) Untrack(vs ...Optimizable) {
	for _, v := range vs {
		delete(o.observed, v)
	}
}

// UntrackAll remove all the tracked parameters.
func (o *GradientDescent) UntrackAll() {
	for v := range o.observed {
		delete(o.observed, v)
	}
}

// Optimize optimize the params, applying the optional gradient clipping.
// After the optimization the params have zero gradients.
func (o *GradientDescent) Optimize() {
	o.clipGrads()
	o.updateParams()
	o.ZeroGrad()
}

// updateParamsSerial applies the optimization method to all the observed parameters.
func (o *GradientDescent) updateParamsSerial() {
	for param := range o.observed {
		if param.HasGrad() {
			delta := o.method.Delta(param) // important: don't release delta here
			param.ApplyDelta(delta)
		}
	}
}

// updateParams applies the optimization method to all the observed parameters concurrently.
// TODO: distribute the workload proportionately to the number of available CPUs?
func (o *GradientDescent) updateParams() {
	var wg sync.WaitGroup
	for key := range o.observed {
		if key.HasGrad() {
			wg.Add(1)
			go func(param Optimizable) {
				defer wg.Done()
				delta := o.method.Delta(param)
				param.ApplyDelta(delta)
			}(key)
		}
	}
	wg.Wait()
}

// clipGrad applies the gradient clipping to all the observed parameters.
func (o *GradientDescent) clipGrads() {
	if o.gradClipper != nil {
		var gs []mat.Matrix
		for param := range o.observed {
			gs = append(gs, param.Grad())
		}
		o.gradClipper.Clip(gs)
	}
}

// ZeroGrad set the gradients of the observed variables to zeros
func (o *GradientDescent) ZeroGrad() {
	for variable := range o.observed {
		variable.ZeroGrad()
	}
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
