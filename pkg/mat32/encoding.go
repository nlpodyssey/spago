// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat32

import (
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"io"
	"math"
)

func init() {
	gob.Register(&Dense{})
	gob.Register(&Sparse{})
}

// MarshalBinary marshals a Dense matrix into binary form.
func (d Dense) MarshalBinary() ([]byte, error) {
	data := make([]byte, 8+d.size*4)
	binary.LittleEndian.PutUint32(data, uint32(d.rows))
	binary.LittleEndian.PutUint32(data[4:], uint32(d.cols))
	for i, v := range d.data {
		binary.LittleEndian.PutUint32(data[8+i*4:], math.Float32bits(v))
	}
	return data, nil
}

// UnmarshalBinary unmarshals a binary representation of a Dense matrix.
func (d *Dense) UnmarshalBinary(data []byte) error {
	d.viewOf = nil
	d.fromPool = false
	d.rows = int(binary.LittleEndian.Uint32(data))
	d.cols = int(binary.LittleEndian.Uint32(data[4:]))
	d.size = d.rows * d.cols
	d.data = make([]Float, d.size)
	for i := range d.data {
		d.data[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[8+i*4:]))
	}
	return nil
}

// MarshalBinary marshals a Sparse matrix into binary form.
func (s Sparse) MarshalBinary() ([]byte, error) {
	data := make([]byte, 8+s.size*4)
	binary.LittleEndian.PutUint32(data, uint32(s.rows))
	binary.LittleEndian.PutUint32(data[4:], uint32(s.cols))
	for i, v := range s.Data() {
		binary.LittleEndian.PutUint32(data[8+i*4:], math.Float32bits(v))
	}
	return data, nil
}

// UnmarshalBinary unmarshals a binary representation of a Sparse matrix.
func (s *Sparse) UnmarshalBinary(data []byte) error {
	rows := int(binary.LittleEndian.Uint32(data))
	cols := int(binary.LittleEndian.Uint32(data[4:]))
	elements := make([]Float, rows*cols)
	for i := range elements {
		elements[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[8+i*4:]))
	}
	*s = *NewSparse(rows, cols, elements)
	return nil
}

const (
	binaryNilMatrix byte = iota
	binaryDenseMatrix
	binarySparseMatrix
)

// MarshalBinaryMatrix encodes a Matrix into binary form.
func MarshalBinaryMatrix(m Matrix, w io.Writer) error {
	var mType byte
	var bin []byte
	var err error

	switch v := m.(type) {
	case nil:
		mType = binaryNilMatrix
	case *Dense:
		mType = binaryDenseMatrix
		bin, err = v.MarshalBinary()
	case *Sparse:
		mType = binarySparseMatrix
		bin, err = v.MarshalBinary()
	default:
		return fmt.Errorf("unknown matrix type %T: %#v", m, m)
	}

	if err != nil {
		return err
	}

	_, err = w.Write([]byte{mType})
	if err != nil {
		return err
	}
	if mType == binaryNilMatrix {
		return nil
	}

	binLen := make([]byte, 4)
	binary.LittleEndian.PutUint32(binLen, uint32(len(bin)))
	_, err = w.Write(binLen)
	if err != nil {
		return err
	}

	_, err = w.Write(bin)
	if err != nil {
		return err
	}

	return nil
}

// UnmarshalBinaryMatrix decodes a Matrix from binary form.
func UnmarshalBinaryMatrix(r io.Reader) (Matrix, error) {
	smType := make([]byte, 1)
	_, err := r.Read(smType)
	if err != nil {
		return nil, err
	}
	mType := smType[0]
	if mType == binaryNilMatrix {
		return nil, nil
	}

	binLenBytes := make([]byte, 4)
	_, err = r.Read(binLenBytes)
	if err != nil {
		return nil, err
	}
	binLen := int(binary.LittleEndian.Uint32(binLenBytes))
	bin := make([]byte, binLen)
	_, err = r.Read(bin)
	if err != nil {
		return nil, err
	}

	switch mType {
	case binaryDenseMatrix:
		m := new(Dense)
		err = m.UnmarshalBinary(bin)
		return m, err
	case binarySparseMatrix:
		m := new(Sparse)
		err = m.UnmarshalBinary(bin)
		return m, err
	default:
		return nil, fmt.Errorf("unknown binary matrix type %d", mType)
	}
}
