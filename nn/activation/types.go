// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package activation

import (
	"fmt"
	"strings"
)

// Name is the enumeration-like type used for the set of built-in activations.
type Name int

const (
	// Identity identifies the Graph.Identity operator.
	Identity Name = iota
	// Tan identifies the Graph.Tan operator.
	Tan
	// Tanh identifies the Graph.Tanh operator.
	Tanh
	// Sigmoid identifies the Graph.Sigmoid operator.
	Sigmoid
	// HardSigmoid identifies the Graph.HardSigmoid operator.
	HardSigmoid
	// HardTanh identifies the Graph.HardTanh operator.
	HardTanh
	// Softsign identifies the Graph.Softsign operator.
	Softsign
	// ReLU identifies the Graph.ReLU operator.
	ReLU
	// CELU identifies the Graph.CELU operator.
	CELU
	// GELU identifies the Graph.GELU operator.
	GELU
	// ELU identifies the Graph.ELU operator.
	ELU
	// PositiveELU identifies the Graph.PositiveELU operator.
	PositiveELU
	// SwishB identifies the Graph.SwishB operator.
	SwishB
	// Swish identifies the Graph.Swish operator.
	Swish
	// SiLU identifies the Graph.SiLU operator.
	SiLU
	// Mish identifies the Graph.Mish operator.
	Mish
	// LeakyReLU identifies the Graph.LeakyReLU operator.
	LeakyReLU
	// SELU identifies the Graph.SELU operator.
	SELU
	// SoftPlus identifies the Graph.SoftPlus operator.
	SoftPlus
	// SoftShrink identifies the Graph.SoftShrink operator.
	SoftShrink
	// Threshold identifies the Graph.Threshold operator.
	Threshold
	// Softmax identifies the Graph.Softmax operator.
	Softmax
	// LogSoftmax identifies the Graph.LogSoftmax operator.
	LogSoftmax
	// SparseMax identifies the Graph.SparseMax operator.
	SparseMax
)

var activationsMap = map[Name]string{
	Identity:    "Identity",
	Tan:         "Tan",
	Tanh:        "Tanh",
	Sigmoid:     "Sigmoid",
	HardSigmoid: "HardSigmoid",
	HardTanh:    "HardTanh",
	Softsign:    "Softsign",
	ReLU:        "ReLU",
	CELU:        "CELU",
	GELU:        "GELU",
	ELU:         "ELU",
	PositiveELU: "PositiveELU",
	SwishB:      "SwishB",
	Swish:       "Swish",
	SiLU:        "SiLU",
	Mish:        "Mish",
	LeakyReLU:   "LeakyReLU",
	SELU:        "SELU",
	SoftPlus:    "SoftPlus",
	SoftShrink:  "SoftShrink",
	Threshold:   "Threshold",
	Softmax:     "Softmax",
	LogSoftmax:  "LogSoftmax",
	SparseMax:   "SparseMax",
}

var strActivationMap = strToActivationMap()

// strToName maps a string to a Name.
func strToActivationMap() map[string]Name {
	invMap := make(map[string]Name)
	for k, v := range activationsMap {
		invMap[v] = k
		invMap[strings.ToLower(v)] = k
	}
	return invMap
}

// Activation maps a string to an activation function.
// It returns an error if the string does not match any built-in activation (not even using lowercase).
func Activation(str string) (Name, error) {
	if value, ok := strActivationMap[str]; ok {
		return value, nil
	}
	return -1, fmt.Errorf("activation: unknown activation function %s", str)
}

// MustActivation maps a string to an activation function.
// It panics if the string does not match any built-in activation (not even using lowercase).
func MustActivation(str string) Name {
	value, err := Activation(str)
	if err != nil {
		panic(err)
	}
	return value
}
