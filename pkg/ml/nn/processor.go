// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import "github.com/nlpodyssey/spago/pkg/ml/ag"

// ProcessingMode regulates the different usage of some operations (e.g. Dropout, BatchNorm, etc.) inside a Processor,
// depending on whether you're doing training or inference.
// Failing to set the right mode will yield inconsistent inference results.
type ProcessingMode int

const (
	Training ProcessingMode = iota
	Inference
)

// Processor performs the operations on the computational graphs using the model's parameters.
type Processor interface {
	// Model returns the model the processor belongs to.
	Model() Model
	// Mode returns whether the processor is being used for training or inference.
	Mode() ProcessingMode
	// SetMode tells the processor to operate in training or inference mode.
	SetMode(mode ProcessingMode)
	// Graph returns the computational graph on which the processor operates.
	Graph() *ag.Graph
	// Whether the processor needs the complete sequence to start processing (as in the case of BiRNN and other bidirectional models), or not.
	RequiresFullSeq() bool
	// Forward performs the the forward step for each input and returns the result.
	// Recurrent networks treats the input nodes as a sequence.
	// Differently, feed-forward networks are stateless so every computation is independent.
	Forward(xs ...ag.Node) []ag.Node
}

// SetProcessingMode sets the processing mode to a group of processors.
func SetProcessingMode(mode ProcessingMode, ps ...Processor) {
	for _, proc := range ps {
		proc.SetMode(mode)
	}
}
