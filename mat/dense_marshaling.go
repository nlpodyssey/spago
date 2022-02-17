// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"math"
)

func init() {
	gob.Register(&Dense[float32]{})
	gob.Register(&Dense[float64]{})
}

// MarshalBinary marshals a Dense matrix into binary form.
func (d Dense[T]) MarshalBinary() ([]byte, error) {
	switch any(T(0)).(type) {
	case float32:
		return d.marshalBinaryFloat32()
	case float64:
		return d.marshalBinaryFloat64()
	default:
		panic(fmt.Sprintf("mat: unexpected dense matrix type %T", T(0)))
	}
}

func (d Dense[T]) marshalBinaryFloat32() ([]byte, error) {
	data := make([]byte, 8+d.Size()*4)
	binary.LittleEndian.PutUint32(data, uint32(d.rows))
	binary.LittleEndian.PutUint32(data[4:], uint32(d.cols))
	for i, v := range d.data {
		binary.LittleEndian.PutUint32(data[8+i*4:], math.Float32bits(float32(v)))
	}
	return data, nil
}

func (d Dense[T]) marshalBinaryFloat64() ([]byte, error) {
	data := make([]byte, 8+d.Size()*8)
	binary.LittleEndian.PutUint32(data, uint32(d.rows))
	binary.LittleEndian.PutUint32(data[4:], uint32(d.cols))
	for i, v := range d.data {
		binary.LittleEndian.PutUint64(data[8+i*8:], math.Float64bits(float64(v)))
	}
	return data, nil
}

// UnmarshalBinary unmarshals a binary representation of a Dense matrix.
func (d *Dense[T]) UnmarshalBinary(data []byte) error {
	switch any(T(0)).(type) {
	case float32:
		return d.unmarshalBinaryFloat32(data)
	case float64:
		return d.unmarshalBinaryFloat64(data)
	default:
		panic(fmt.Sprintf("mat: unexpected dense matrix type %T", T(0)))
	}
}

func (d *Dense[T]) unmarshalBinaryFloat32(data []byte) error {
	d.flags = 0
	d.rows = int(binary.LittleEndian.Uint32(data))
	d.cols = int(binary.LittleEndian.Uint32(data[4:]))
	size := d.rows * d.cols
	d.data = make([]T, size)

	if expected := 8 + size*4; len(data) != expected {
		return fmt.Errorf("mat: cannot unmarshal float32 Dense matrix. Data len expected %d, actual %d", expected, len(data))
	}
	for i := range d.data {
		d.data[i] = T(math.Float32frombits(binary.LittleEndian.Uint32(data[8+i*4:])))
	}
	return nil
}

func (d *Dense[T]) unmarshalBinaryFloat64(data []byte) error {
	d.flags = 0
	d.rows = int(binary.LittleEndian.Uint32(data))
	d.cols = int(binary.LittleEndian.Uint32(data[4:]))
	size := d.rows * d.cols
	d.data = make([]T, size)

	if expected := 8 + size*8; len(data) != expected {
		return fmt.Errorf("mat: cannot unmarshal float64 Dense matrix. Data len expected %d, actual %d", expected, len(data))
	}
	for i := range d.data {
		d.data[i] = T(math.Float64frombits(binary.LittleEndian.Uint64(data[8+i*8:])))
	}
	return nil
}
