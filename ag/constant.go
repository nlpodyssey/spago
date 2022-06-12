// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"bytes"
	"encoding/gob"
	"fmt"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
)

// Constant is a type of Node that can only handle a value, but not gradients.
type Constant struct {
	value mat.Matrix
	name  string
}

// Const creates a new Constant Node.
func Const(value mat.Matrix) *Constant {
	return &Constant{
		value: value,
	}
}

// ScalarConst creates a new Constant from a scalar value.
func ScalarConst[T float.DType](value T) *Constant {
	return &Constant{
		value: mat.NewScalar(value),
	}
}

// WithName sets the Constant's name.
func (c *Constant) WithName(value string) *Constant {
	c.name = value
	return c
}

// Name returns the Name of the Constant (it can be empty).
// If a constant has no name, and the value is a scalar, then it returns its value.
//
// Identifying a Constant solely upon its name is highly discouraged.
// The name should be used solely for debugging or testing purposes.
func (c *Constant) Name() string {
	if c.name != "" {
		return c.name
	}
	if mat.IsScalar(c.value) {
		return fmt.Sprint(c.value.Scalar())
	}
	return c.name
}

// Value returns the value of the constant itself.
func (c *Constant) Value() mat.Matrix {
	return c.value
}

// Grad satisfies the Node interface, and always returns nil for a Constant.
func (c *Constant) Grad() mat.Matrix {
	return nil
}

// AccGrad satisfies the Node interface, and is a no-op for a Constant.
func (c *Constant) AccGrad(mat.Matrix) {}

// HasGrad satisfies the Node interface, and always returns false for a Constant.
func (c *Constant) HasGrad() bool {
	return false
}

// RequiresGrad satisfies the Node interface, and always returns false for a Constant.
func (c *Constant) RequiresGrad() bool {
	return false
}

// ZeroGrad satisfies the Node interface, and is a no-op for a Constant.
func (c *Constant) ZeroGrad() {}

// MarshalBinary marshals a Constant into binary form.
func (c *Constant) MarshalBinary() ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	err := enc.Encode(struct {
		Value mat.Matrix
		Name  string
	}{
		Value: c.value,
		Name:  c.name,
	})
	if err != nil {
		return nil, fmt.Errorf("cannot encode Constant: %w", err)
	}
	return buf.Bytes(), nil
}

// UnmarshalBinary unmarshals a Constant from binary form.
func (c *Constant) UnmarshalBinary(data []byte) error {
	r := bytes.NewReader(data)
	dec := gob.NewDecoder(r)
	var v struct {
		Value mat.Matrix
		Name  string
	}
	err := dec.Decode(&v)
	if err != nil {
		return fmt.Errorf("cannot decode Constant: %w", err)
	}
	c.value = v.Value
	c.name = v.Name
	return nil
}
