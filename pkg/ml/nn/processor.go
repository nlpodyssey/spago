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
	// GetModel returns the model the processor belongs to.
	GetModel() Model
	// GetMode returns whether the processor is being used for training or inference.
	GetMode() ProcessingMode
	// SetMode tells the processor to operate in training or inference mode.
	SetMode(mode ProcessingMode)
	// GetGraph returns the computational graph on which the processor operates.
	GetGraph() *ag.Graph
	// RequiresFullSeq returns whether the processor needs the complete sequence to start processing
	// (as in the case of BiRNN and other bidirectional models), or not.
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

// BaseProcessors satisfies some methods of the Processor interface.
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

// SetMode tells the processor to operate in training or inference mode.
// It must be overridden whenever the processor includes sub-processors.
func (p *BaseProcessor) SetMode(mode ProcessingMode) {
	p.Mode = mode
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
