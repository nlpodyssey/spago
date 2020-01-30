// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gd

import (
	"bytes"
	"encoding/binary"
	"io"
	"saientist.dev/spago/pkg/mat"
	"saientist.dev/spago/pkg/utils"
)

// MarshalBinaryTo returns the number of bytes written into w and an error, if any.
func MarshalBinaryTo(supp *Support, w io.Writer) (int, error) {
	h := header{Opt: int64(supp.Name), SuppSize: int64(len(supp.Data))}
	n, err := h.marshalBinaryTo(w)
	if err != nil {
		return n, err
	}
	nn, err := mat.MarshalBinarySlice(supp.Data, w)
	n += nn
	return n, err
}

//
func NewUnmarshalBinaryFrom(r io.Reader) (*Support, int, error) {
	var h header
	n, err := h.unmarshalBinaryFrom(r)
	if err != nil {
		return nil, n, err
	}
	data := make([]mat.Matrix, h.SuppSize)
	nn, err := mat.NewUnmarshalBinarySlice(data, r)
	n = +nn
	if err != nil {
		return nil, n, err
	}
	supp := &Support{
		Name: MethodName(h.Opt),
		Data: data,
	}
	return supp, n, err
}

type header struct {
	Opt      int64
	SuppSize int64
}

var headerSize = binary.Size(header{})

func (s header) marshalBinaryTo(w io.Writer) (int, error) {
	buf := bytes.NewBuffer(make([]byte, 0, headerSize))
	err := binary.Write(buf, binary.LittleEndian, s)
	if err != nil {
		return 0, err
	}
	return w.Write(buf.Bytes())
}

func (s *header) unmarshalBinary(buf []byte) error {
	err := binary.Read(bytes.NewReader(buf), binary.LittleEndian, s)
	if err != nil {
		return err
	}
	return nil
}

func (s *header) unmarshalBinaryFrom(r io.Reader) (int, error) {
	buf := make([]byte, headerSize)
	n, err := utils.ReadFull(r, buf)
	if err != nil {
		return n, err
	}
	return n, s.unmarshalBinary(buf[:n])
}
