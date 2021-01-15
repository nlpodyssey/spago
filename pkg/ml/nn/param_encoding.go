// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"bytes"
	"encoding/binary"
	"encoding/gob"
	"fmt"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"io"
	"io/ioutil"
	"log"
)

// init registers the param implementation with the gob subsystem - so that it knows how to encode and decode
// values of type nn.Param
func init() {
	gob.Register(&param{})
}

// MarshalBinary marshals a param into binary form.
func (r *param) MarshalBinary() ([]byte, error) {
	buf := new(bytes.Buffer)

	err := mat.MarshalBinaryMatrix(r.value, buf)
	if err != nil {
		return nil, err
	}

	if r.payload == nil {
		buf.WriteByte(0)
	} else {
		buf.WriteByte(1)
		pBin, err := r.payload.MarshalBinary()
		if err != nil {
			return nil, err
		}
		buf.Write(pBin)
	}

	return buf.Bytes(), nil
}

// UnmarshalBinary unmarshals a param from binary form.
func (r *param) UnmarshalBinary(data []byte) error {
	var err error
	buf := bytes.NewReader(data)

	r.value, err = mat.UnmarshalBinaryMatrix(buf)
	if err != nil {
		return err
	}

	hasPayload, err := buf.ReadByte()
	if hasPayload == 0 {
		r.payload = nil
		return nil
	}

	r.payload = new(Payload)

	pBin, err := ioutil.ReadAll(buf)
	if err != nil {
		return err
	}

	return r.payload.UnmarshalBinary(pBin)
}

// MarshalBinaryParam encodes a Param into binary form.
func MarshalBinaryParam(p Param, w io.Writer) error {
	if p == nil {
		_, err := w.Write([]byte{0})
		return err
	}

	pp, isParam := p.(*param)
	if !isParam {
		log.Fatal(fmt.Errorf("unsupported Param implementation for binary marshaling, %T: %#v", p, p))
	}

	_, err := w.Write([]byte{1})
	if err != nil {
		return err
	}

	bin, err := pp.MarshalBinary()
	if err != nil {
		return err
	}

	binLen := make([]byte, 4)
	binary.LittleEndian.PutUint32(binLen, uint32(len(bin)))
	_, err = w.Write(binLen)
	if err != nil {
		return err
	}

	_, err = w.Write(bin)
	return err
}

// UnmarshalBinaryParam decodes a Param from binary form.
// TODO: add a "withBacking" optional argument to remove the need of UnmarshalBinaryParamWithReceiver().
func UnmarshalBinaryParam(r io.Reader) (Param, error) {
	isPresent := make([]byte, 1)
	_, err := r.Read(isPresent)
	if err != nil {
		return nil, err
	}
	if isPresent[0] == 0 {
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

	p := new(param)
	err = p.UnmarshalBinary(bin)
	if err != nil {
		return nil, err
	}
	return p, nil
}

// UnmarshalBinaryParamWithReceiver decodes a Param from binary form into the receiver.
func UnmarshalBinaryParamWithReceiver(r io.Reader, dest Param) error {
	p, isParam := dest.(*param)
	if !isParam {
		log.Fatal(fmt.Errorf("unsupported Param implementation for binary unmarshaling, %T: %#v", p, p))
	}

	isPresent := make([]byte, 1)
	_, err := r.Read(isPresent)
	if err != nil {
		return err
	}
	if isPresent[0] == 0 {
		return nil
	}

	binLenBytes := make([]byte, 4)
	_, err = r.Read(binLenBytes)
	if err != nil {
		return err
	}
	binLen := int(binary.LittleEndian.Uint32(binLenBytes))
	bin := make([]byte, binLen)
	_, err = r.Read(bin)
	if err != nil {
		return err
	}

	err = p.UnmarshalBinary(bin)
	if err != nil {
		return err
	}
	return nil
}
