// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package activation

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/nlpodyssey/spago/ag"
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

var (
	strActivationMap = strToActivationMap()
	activationsMap   = map[Name]strOperatorPair{
		Identity:    {str: "Identity", operator: reflect.ValueOf(ag.Identity)},
		Tan:         {str: "Tan", operator: reflect.ValueOf(ag.Tan)},
		Tanh:        {str: "Tanh", operator: reflect.ValueOf(ag.Tanh)},
		Sigmoid:     {str: "Sigmoid", operator: reflect.ValueOf(ag.Sigmoid)},
		HardSigmoid: {str: "HardSigmoid", operator: reflect.ValueOf(ag.HardSigmoid)},
		HardTanh:    {str: "HardTanh", operator: reflect.ValueOf(ag.HardTanh)},
		Softsign:    {str: "Softsign", operator: reflect.ValueOf(ag.Softsign)},
		ReLU:        {str: "ReLU", operator: reflect.ValueOf(ag.ReLU)},
		CELU:        {str: "CELU", operator: reflect.ValueOf(ag.CELU)},
		GELU:        {str: "GELU", operator: reflect.ValueOf(ag.GELU)},
		ELU:         {str: "ELU", operator: reflect.ValueOf(ag.ELU)},
		PositiveELU: {str: "PositiveELU", operator: reflect.ValueOf(ag.PositiveELU)},
		SwishB:      {str: "SwishB", operator: reflect.ValueOf(ag.SwishB)},
		Swish:       {str: "Swish", operator: reflect.ValueOf(ag.Swish)},
		SiLU:        {str: "SiLU", operator: reflect.ValueOf(ag.SiLU)},
		Mish:        {str: "Mish", operator: reflect.ValueOf(ag.Mish)},
		LeakyReLU:   {str: "LeakyReLU", operator: reflect.ValueOf(ag.LeakyReLU)},
		SELU:        {str: "SELU", operator: reflect.ValueOf(ag.SELU)},
		SoftPlus:    {str: "SoftPlus", operator: reflect.ValueOf(ag.SoftPlus)},
		SoftShrink:  {str: "SoftShrink", operator: reflect.ValueOf(ag.SoftShrink)},
		Threshold:   {str: "Threshold", operator: reflect.ValueOf(ag.Threshold)},
		Softmax:     {str: "Softmax", operator: reflect.ValueOf(ag.Softmax)},
		LogSoftmax:  {str: "LogSoftmax", operator: reflect.ValueOf(ag.LogSoftmax)},
		SparseMax:   {str: "SparseMax", operator: reflect.ValueOf(ag.SparseMax)},
	}
)

type strOperatorPair struct {
	str      string
	operator reflect.Value
}

// strToName maps a string to a Name.
func strToActivationMap() map[string]Name {
	invMap := make(map[string]Name)
	for k, v := range activationsMap {
		invMap[v.str] = k
		invMap[strings.ToLower(v.str)] = k
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

// Do make a new node as a result of the application of the input operator.
func Do(act Name, xs ...ag.DualValue) ag.DualValue {
	v := activationsMap[act].operator
	args := make([]reflect.Value, len(xs))
	for i, x := range xs {
		args[i] = reflect.ValueOf(x)
	}
	ret := v.Call(args)
	return ret[0].Interface().(ag.DualValue)
}
