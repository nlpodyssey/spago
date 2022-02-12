// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"fmt"
	"strings"
)

type moduleFieldType uint8

const (
	defaultModuleFieldType moduleFieldType = iota
	paramsModuleFieldType
	weightsModuleFieldType
	biasesModuleFieldType
	undefinedModuleFieldType
)

type moduleFieldScope uint8

const (
	defaultModuleFieldScope moduleFieldScope = iota
	processorModuleFieldScope
	modelModuleFieldScope
)

// ParamsType is the enumeration-like type used for the set of parameter
// (Param) types of a neural network Model.
type ParamsType uint8

const (
	// Weights identifies a Param containing weights.
	Weights ParamsType = iota
	// Biases identifies a Param containing biases.
	Biases
	// Undefined identifies a generic Param, which cannot be described
	// with other ParamsType values.
	Undefined
)

func (t ParamsType) String() string {
	return [...]string{"weights", "biases", "undefined"}[t] // important lower case
}

type moduleFieldTag struct {
	Type  moduleFieldType
	Scope moduleFieldScope
}

func parseModuleFieldTag(tag string) (moduleFieldTag, error) {
	mft := moduleFieldTag{
		Type:  defaultModuleFieldType,
		Scope: defaultModuleFieldScope,
	}
	if len(tag) == 0 {
		return mft, nil
	}
	var err error
	for _, token := range strings.Split(tag, ";") {
		sepIndex := strings.Index(token, ":")
		if sepIndex == -1 {
			return mft, fmt.Errorf("malformed module field tag %#v: `hey:value` form expected", tag)
		}
		left, right := token[:sepIndex], token[sepIndex+1:]
		switch left {
		case "type":
			mft.Type, err = stringToModuleFieldType(right)
			if err != nil {
				return mft, fmt.Errorf("malformed module field tag %#v: %w", tag, err)
			}
		case "scope":
			mft.Scope, err = stringToModuleFieldScope(right)
			if err != nil {
				return mft, fmt.Errorf("malformed module field tag %#v: %w", tag, err)
			}
		default:
			return mft, fmt.Errorf("malformed module field tag %#v: unexpected key %#v", tag, left)
		}
	}
	return mft, nil
}

func stringToModuleFieldType(s string) (moduleFieldType, error) {
	switch s {
	case "params":
		return paramsModuleFieldType, nil
	case "weights":
		return weightsModuleFieldType, nil
	case "biases":
		return biasesModuleFieldType, nil
	case "undefined":
		return undefinedModuleFieldType, nil
	default:
		return defaultModuleFieldType, fmt.Errorf("unexpected model field type %#v", s)
	}
}

func stringToModuleFieldScope(s string) (moduleFieldScope, error) {
	switch s {
	case "processor":
		return processorModuleFieldScope, nil
	case "model":
		return modelModuleFieldScope, nil
	default:
		return defaultModuleFieldScope, fmt.Errorf("unexpected model field scope %#v", s)
	}
}

func (m *moduleFieldTag) paramType() ParamsType {
	switch m.Type {
	case weightsModuleFieldType:
		return Weights
	case biasesModuleFieldType:
		return Biases
	default:
		return Undefined
	}
}
