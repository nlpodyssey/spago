// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package activation

import (
	"fmt"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"reflect"
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

var (
	activationsFloat32 = activations[float32]()
	activationsFloat64 = activations[float64]()
	strActivationMap   = strToActivationMap()
)

type strOperatorPair struct {
	str      string
	operator reflect.Value
}

func activations[T mat.DType]() map[Name]strOperatorPair {
	return map[Name]strOperatorPair{
		Identity:    {str: "Identity", operator: reflect.ValueOf(ag.Identity[T])},
		Tan:         {str: "Tan", operator: reflect.ValueOf(ag.Tan[T])},
		Tanh:        {str: "Tanh", operator: reflect.ValueOf(ag.Tanh[T])},
		Sigmoid:     {str: "Sigmoid", operator: reflect.ValueOf(ag.Sigmoid[T])},
		HardSigmoid: {str: "HardSigmoid", operator: reflect.ValueOf(ag.HardSigmoid[T])},
		HardTanh:    {str: "HardTanh", operator: reflect.ValueOf(ag.HardTanh[T])},
		Softsign:    {str: "Softsign", operator: reflect.ValueOf(ag.Softsign[T])},
		ReLU:        {str: "ReLU", operator: reflect.ValueOf(ag.ReLU[T])},
		CELU:        {str: "CELU", operator: reflect.ValueOf(ag.CELU[T])},
		GELU:        {str: "GELU", operator: reflect.ValueOf(ag.GELU[T])},
		ELU:         {str: "ELU", operator: reflect.ValueOf(ag.ELU[T])},
		PositiveELU: {str: "PositiveELU", operator: reflect.ValueOf(ag.PositiveELU[T])},
		SwishB:      {str: "SwishB", operator: reflect.ValueOf(ag.SwishB[T])},
		Swish:       {str: "Swish", operator: reflect.ValueOf(ag.Swish[T])},
		SiLU:        {str: "SiLU", operator: reflect.ValueOf(ag.SiLU[T])},
		Mish:        {str: "Mish", operator: reflect.ValueOf(ag.Mish[T])},
		LeakyReLU:   {str: "LeakyReLU", operator: reflect.ValueOf(ag.LeakyReLU[T])},
		SELU:        {str: "SELU", operator: reflect.ValueOf(ag.SELU[T])},
		SoftPlus:    {str: "SoftPlus", operator: reflect.ValueOf(ag.SoftPlus[T])},
		SoftShrink:  {str: "SoftShrink", operator: reflect.ValueOf(ag.SoftShrink[T])},
		Threshold:   {str: "Threshold", operator: reflect.ValueOf(ag.Threshold[T])},
		Softmax:     {str: "Softmax", operator: reflect.ValueOf(ag.Softmax[T])},
		LogSoftmax:  {str: "LogSoftmax", operator: reflect.ValueOf(ag.LogSoftmax[T])},
		SparseMax:   {str: "SparseMax", operator: reflect.ValueOf(ag.SparseMax[T])},
	}
}

// strToName maps a string to a Name.
func strToActivationMap() map[string]Name {
	invMap := make(map[string]Name)
	for k, v := range activationsFloat32 {
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
func Do[T mat.DType](act Name, xs ...ag.Node[T]) ag.Node[T] {
	v := activationsForFloat[T]()[act].operator
	args := make([]reflect.Value, len(xs))
	for i, x := range xs {
		args[i] = reflect.ValueOf(x)
	}
	ret := v.Call(args)
	return ret[0].Interface().(ag.Node[T])
}

func activationsForFloat[T mat.DType]() map[Name]strOperatorPair {
	switch any(T(0)).(type) {
	case float32:
		return activationsFloat32
	case float64:
		return activationsFloat64
	default:
		panic(fmt.Sprintf("activation: invalid type %T", T(0)))
	}
}
