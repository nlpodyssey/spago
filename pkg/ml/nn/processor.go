// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
)

// ProcessingMode regulates the different usage of some operations (e.g. Dropout, BatchNorm, etc.) inside a Processor,
// depending on whether you're doing training or inference.
// Failing to set the right mode will yield inconsistent inference results.
type ProcessingMode int

const (
	// Training is to be used during the training phase of a model. For example, dropouts are enabled.
	Training ProcessingMode = iota
	// Inference keeps weights fixed while using the model and disables some operations (e.g. skip dropout).
	Inference
)

// Context is used to instantiate a processor to operate on a graph, according to the desired ProcessingMode.
// If a processor contains other sub-processors, you must instantiate them using the same context to make sure
// you are operating on the same graph and in the same mode.
type Context struct {
	// Graph is the computational graph on which the processor(s) operate.
	Graph *ag.Graph
	// Mode regulates the different usage of some operations whether you're doing training or inference.
	Mode ProcessingMode
}

// Processor performs the operations on the computational graphs using the model's parameters.
type Processor interface {
	// GetModel returns the model the processor belongs to.
	GetModel() Model
	// GetMode returns whether the processor is being used for training or inference.
	GetMode() ProcessingMode
	// GetGraph returns the computational graph on which the processor operates.
	GetGraph() *ag.Graph
	// RequiresFullSeq returns whether the processor needs the complete sequence to start processing
	// (as in the case of BiRNN and other bidirectional models), or not.
	RequiresFullSeq() bool
	// Forward performs the forward step for each input and returns the result.
	// Recurrent networks treats the input nodes as a sequence.
	// Differently, feed-forward networks are stateless so every computation is independent.
	Forward(xs ...ag.Node) []ag.Node
}

// BaseProcessor satisfies some methods of the Processor interface.
// It is meant to be embedded in other processors to reduce the amount of boilerplate code.
type BaseProcessor struct {
	Model             Model
	Mode              ProcessingMode
	Graph             *ag.Graph
	FullSeqProcessing bool
}

// GetModel returns the model the processor belongs to.
func (p *BaseProcessor) GetModel() Model {
	return p.Model
}

// GetMode returns whether the processor is being used for training or inference.
func (p *BaseProcessor) GetMode() ProcessingMode {
	return p.Mode
}

// GetGraph returns the computational graph on which the processor operates.
func (p *BaseProcessor) GetGraph() *ag.Graph {
	return p.Graph
}

// RequiresFullSeq returns whether the processor needs the complete sequence to start processing
// (as in the case of BiRNN and other bidirectional models), or not.
func (p *BaseProcessor) RequiresFullSeq() bool {
	return p.FullSeqProcessing
}
