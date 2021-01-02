// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gd

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd/clipper"
	"github.com/nlpodyssey/spago/pkg/utils/processingqueue"
	"runtime"
	"sync"
)

// GradientDescent implements Gradients Descent (GD) optimization.
type GradientDescent struct {
	method           Method // optimization method (SGD, AdaGrad, Adam, ...)
	gradClipper      clipper.GradClipper
	paramsGetter     nn.ParamsGetter
	paramsToOptimize []nn.Param
	// processingQueue allows proper handling for computationally heavy operations
	// such as the params update step.
	// The default size is defaultProcessingQueueSize.
	processingQueue processingqueue.ProcessingQueue
}

// defaultProcessingQueueSize is the default size of GradientDescent.processingQueue on a new optimizer.
var defaultProcessingQueueSize = runtime.NumCPU()

// Option allows to configure a new GradientDescent with your specific needs.
type Option func(*GradientDescent)

// ClipGradByValue is an option to clip the gradients during the training between
// -value and +value.
func ClipGradByValue(value mat.Float) Option {
	return func(f *GradientDescent) {
		f.gradClipper = &clipper.ClipValue{Value: value}
	}
}

// ClipGradByNorm is an option to clip the gradients during the training by norm.
func ClipGradByNorm(max, normType mat.Float) Option {
	return func(f *GradientDescent) {
		f.gradClipper = &clipper.ClipNorm{
			MaxNorm:  max,
			NormType: normType,
		}
	}
}

// ConcurrentComputations sets the maximum number of concurrent computations handled by the GradientDescent
// for heavy tasks such as the params update steps.
// The value 1 corresponds to sequential execution.
func ConcurrentComputations(value int) Option {
	if value < 1 {
		panic("gd: ConcurrentComputations value must be greater than zero")
	}
	return func(f *GradientDescent) {
		f.processingQueue = processingqueue.New(value)
	}
}

// NewOptimizer returns a new GradientDescent optimizer. The gradient clipper can be set to nil.
func NewOptimizer(method Method, paramsIterator nn.ParamsGetter, opts ...Option) *GradientDescent {
	optimizer := &GradientDescent{
		method:           method,
		paramsGetter:     paramsIterator,
		paramsToOptimize: make([]nn.Param, 0),
		processingQueue:  processingqueue.New(defaultProcessingQueueSize),
	}
	for _, opt := range opts {
		opt(optimizer)
	}
	return optimizer
}

// Optimize optimize the params, applying the optional gradient clipping.
// After the optimization the params have zero gradients.
func (o *GradientDescent) Optimize() {
	o.paramsToOptimize = o.paramsGetter.Params()
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
func (o *GradientDescent) updateParams() {
	var wg sync.WaitGroup
	for _, param := range o.paramsToOptimize {
		if !param.HasGrad() {
			continue
		}
		wg.Add(1)
		go func(param nn.Param) {
			defer wg.Done()
			o.processingQueue.Run(func() {
				delta := o.method.Delta(param)
				param.ApplyDelta(delta)
			})
			param.ZeroGrad()
		}(param)
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
