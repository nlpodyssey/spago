// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"bytes"
	"encoding/gob"
	"fmt"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
)

// Buffer is a type of Node that do not require gradients but that can be serialized as parameters.
// This is useful e.g. to store constants, to track the mean and std in batch norm layers etc.
type Buffer struct {
	mat.Matrix
	name string
}

// Buf creates a new Buffer Node.
func Buf(value mat.Matrix) *Buffer {
	return &Buffer{
		Matrix: value,
	}
}

// Const creates a new Buffer from a scalar value.
func Const[T float.DType](value T) *Buffer {
	return &Buffer{
		Matrix: mat.NewScalar(value),
	}
}

// WithName sets the Buffer's name.
func (c *Buffer) WithName(value string) *Buffer {
	c.name = value
	return c
}

// Name returns the Name of the Buffer (it can be empty).
// If a constant has no name, and the value is a scalar, then it returns its value.
//
// Identifying a Buffer solely upon its name is highly discouraged.
// The name should be used solely for debugging or testing purposes.
func (c *Buffer) Name() string {
	if c.name != "" {
		return c.name
	}
	if mat.IsScalar(c.Matrix) {
		return fmt.Sprint(c.Matrix.Scalar())
	}
	return c.name
}

// Value returns the value of the constant itself.
func (c *Buffer) Value() mat.Matrix {
	return c.Matrix
}

// Grad satisfies the Node interface, and always returns nil for a Buffer.
func (c *Buffer) Grad() mat.Matrix {
	return nil
}

// AccGrad satisfies the Node interface, and is a no-op for a Buffer.
func (c *Buffer) AccGrad(mat.Matrix) {}

// HasGrad satisfies the Node interface, and always returns false for a Buffer.
func (c *Buffer) HasGrad() bool {
	return false
}

// RequiresGrad satisfies the Node interface, and always returns false for a Buffer.
func (c *Buffer) RequiresGrad() bool {
	return false
}

// ZeroGrad satisfies the Node interface, and is a no-op for a Buffer.
func (c *Buffer) ZeroGrad() {}

// MarshalBinary marshals a Buffer into binary form.
func (c *Buffer) MarshalBinary() ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	err := enc.Encode(struct {
		Value mat.Matrix
		Name  string
	}{
		Value: c.Matrix,
		Name:  c.name,
	})
	if err != nil {
		return nil, fmt.Errorf("cannot encode Buffer: %w", err)
	}
	return buf.Bytes(), nil
}

// UnmarshalBinary unmarshals a Buffer from binary form.
func (c *Buffer) UnmarshalBinary(data []byte) error {
	r := bytes.NewReader(data)
	dec := gob.NewDecoder(r)
	var v struct {
		Value mat.Matrix
		Name  string
	}
	err := dec.Decode(&v)
	if err != nil {
		return fmt.Errorf("cannot decode Buffer: %w", err)
	}
	c.Matrix = v.Value
	c.name = v.Name
	return nil
}
