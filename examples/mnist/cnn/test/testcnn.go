// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"github.com/nlpodyssey/spago/examples/mnist/internal/mnist"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/utils"
	"github.com/nlpodyssey/spago/third_party/GoMNIST"
	"os"
)

func main() {
	modelPath := os.Args[1]

	var datasetPath string
	if len(os.Args) > 2 {
		datasetPath = os.Args[2]
	} else {
		// assuming default path
		datasetPath = "third_party/GoMNIST/data"
	}

	_, testSet, err := GoMNIST.Load(datasetPath)
	if err != nil {
		panic("Error reading MNIST data.")
	}

	// new model initialized with zeros
	model := mnist.NewCNN(
		9,   // kernelSizeX
		9,   // kernelSizeY
		1,   // inputChannels
		10,  // outputChannels
		5,   // maxPoolingRows
		5,   // maxPoolingCols
		160, // hidden
		10,  // out
		ag.ReLU,
		ag.Softmax,
	)
	err = utils.DeserializeFromFile(modelPath, model)
	if err != nil {
		panic("mnist: error during model deserialization.")
	}

	precision := mnist.NewEvaluator(model).Evaluate(mnist.Dataset{
		Set:              testSet,
		FeaturesAsVector: false, // the CNN input is a 28x28 matrix
	}).Precision()
	fmt.Printf("Accuracy: %.2f\n", 100*precision)
}
