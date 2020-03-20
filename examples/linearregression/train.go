// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Example on how to build a simple linear regression model trained with a very basic linear equation i.e. y=2x+1.
*/
package main

import (
	"fmt"
	"github.com/saientist/spago/pkg/ml/ag"
	"github.com/saientist/spago/pkg/ml/initializers"
	"github.com/saientist/spago/pkg/ml/losses"
	"github.com/saientist/spago/pkg/ml/nn"
	"github.com/saientist/spago/pkg/ml/optimizers/gd"
	"github.com/saientist/spago/pkg/ml/optimizers/gd/sgd"
	"golang.org/x/exp/rand"
)

func main() {
	inputDim := 1  // takes variable 'x'
	outputDim := 1 // takes variable 'y'
	learningRate := 0.0001
	epochs := 100
	seed := 734 // seed for random params initialization
	model := NewLinearRegression(inputDim, outputDim)
	criterion := losses.MSESeq                                  // mean squared error
	updater := sgd.New(sgd.NewConfig(learningRate, 0.0, false)) // stochastic gradient descent (no momentum etc.)
	optimizer := gd.NewOptimizer(updater, nil)                  // no gradient clipping
	nn.TrackParams(model, optimizer)                            // link the model to the optimizer

	// Random params initialization
	model.ForEachParam(func(param *nn.Param) {
		if param.Type() == nn.Weights {
			initializers.XavierUniform(param.Value(), 1.0, rand.NewSource(uint64(seed)))
		}
	})

	// Create dummy data for training with a very basic linear equation i.e., y=2x+1.
	// Here, ‘x’ is the independent variable and y is the dependent variable.
	n := 110
	xValues := make([]float64, n)
	yValues := make([]float64, n)
	for i := 0; i < n; i++ {
		xValues[i] = float64(i)
		yValues[i] = 2*xValues[i] + 1
	}

	for epoch := 0; epoch < epochs; epoch++ {
		// you can beats the occurrence of a new epoch e.g. for learning rate annealing
		optimizer.IncEpoch()

		// get a new computational graph (cg)
		cg := ag.NewGraph()

		// Converting x and y values to graph nodes
		var inputs []ag.Node
		var labels []ag.Node
		for i := 0; i < n; i++ {
			inputs = append(inputs, cg.NewScalar(xValues[i]))
			labels = append(labels, cg.NewScalar(yValues[i]))
		}

		// Clear gradient buffers because we don't want any gradient from previous epoch to carry forward.
		// Actually it would not be necessary here because at each optimization the gradients are automatically set to zero.
		optimizer.ZeroGrad()

		// get output (i.e. prediction) from the model, given the inputs
		outputs := model.NewProc(cg).Forward(inputs...)

		// get loss for the predicted output
		loss := criterion(cg, outputs, labels, true) // true = reduce mean

		// get gradients w.r.t to parameters
		cg.Backward(loss)

		// update parameters
		optimizer.Optimize()

		fmt.Printf("epoch %d, loss %.6f\n", epoch, loss.ScalarValue())
	}

	model.ForEachParam(func(param *nn.Param) {
		fmt.Printf("Learned coefficient: %f", param.ScalarValue())
	})

}
