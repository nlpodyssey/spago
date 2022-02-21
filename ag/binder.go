// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"reflect"

	"github.com/nlpodyssey/spago/mat"
)

// NodeBinder allows any type which satisfies the Node interface
// to create a bound version of itself.
type NodeBinder[T mat.DType] interface {
	Bind(g *Graph[T]) Node[T]
}

// Bind returns a new structure of the same type as the one in input
// in which the fields of type Node (including those from other differentiable
// submodules) are connected to the given graph.
func Bind[T mat.DType, D Differentiable[T]](g *Graph[T], i D) D {
	b := &binder[T]{g: g}
	return b.bindStruct(i).(Differentiable[T]).(D)
}

type binder[T mat.DType] struct {
	g *Graph[T]
}

func (r *binder[_]) bindStruct(rawSource interface{}) interface{} {
	source := reflect.ValueOf(rawSource)
	sourceType := reflect.TypeOf(rawSource)

	sourceIsPointer := source.Kind() == reflect.Ptr
	if sourceIsPointer {
		sourceIsPointer = true
		source = source.Elem()
		sourceType = sourceType.Elem()
	}

	destPointer := reflect.New(sourceType)
	dest := destPointer.Elem()

	numFields := source.NumField()
	for fieldIndex := 0; fieldIndex < numFields; fieldIndex++ {
		sourceField, destField := source.Field(fieldIndex), dest.Field(fieldIndex)
		if !sourceField.CanInterface() {
			continue // skip any initialization
		}
		r.bindStructField(sourceField, destField)
	}

	if !sourceIsPointer {
		return dest.Interface()
	}
	return destPointer.Interface()
}

func (r *binder[T]) bindStructField(sourceField, destField reflect.Value) {
	switch sourceFieldT := sourceField.Interface().(type) {
	case *Graph[T]:
		destField.Set(reflect.ValueOf(r.g))
	case Node[T]:
		// This Node MUST be Bindable, otherwise it panics
		destField.Set(reflect.ValueOf(sourceFieldT.(NodeBinder[T]).Bind(r.g)))
	case Differentiable[T]:
		destField.Set(reflect.ValueOf(r.bindDifferentiable(sourceFieldT)))
	default:
		switch sourceField.Kind() {
		case reflect.Slice:
			destField.Set(r.bindSlice(sourceField))
		case reflect.Map:
			destField.Set(r.bindMap(sourceField)) // , tag
		case reflect.Struct, reflect.Ptr:
			if sourceField.Kind() == reflect.Ptr && sourceField.IsNil() {
				return
			}
			destField.Set(sourceField)
		default:
			destField.Set(sourceField)
		}
	}
}

func (r *binder[T]) bindDifferentiable(sourceField Differentiable[T]) Differentiable[T] {
	if isNil(sourceField) {
		return sourceField
	}
	return Bind(r.g, sourceField).(Differentiable[T])
}

func (r *binder[T]) bindDifferentiableSlice(sourceField []Differentiable[T]) []Differentiable[T] {
	result := make([]Differentiable[T], len(sourceField))
	for i := 0; i < len(sourceField); i++ {
		if isNil(sourceField) {
			return sourceField
		}
		result[i] = Bind(r.g, sourceField[i]).(Differentiable[T])
	}
	return result
}

func (r *binder[T]) bindSlice(sourceField reflect.Value) reflect.Value {
	length := sourceField.Len()
	result := reflect.MakeSlice(sourceField.Type(), length, length)

	for i := 0; i < length; i++ {
		sourceItem := sourceField.Index(i)
		switch sourceItem.Kind() {
		case reflect.Struct, reflect.Ptr, reflect.Interface:
			_, isDifferentiable := sourceItem.Interface().(Differentiable[T])
			_, isNode := sourceItem.Interface().(Node[T])
			if isDifferentiable || isNode {
				r.bindStructField(sourceItem, result.Index(i))
			}
		default:
			return sourceField
		}
	}
	return result
}

func paramInterfaceNamePrefix[T mat.DType]() string {
	return reflect.TypeOf(Node[T](nil)).Elem().Name() // TODO: fix this (?)
}

func (r *binder[T]) bindMap(sourceValue reflect.Value) reflect.Value {
	sourceType := reflect.TypeOf(sourceValue.Interface())
	mapValueType := sourceType.Elem()

	//if mapValueType.Name() != paramInterfaceNamePrefix[T]() { // TODO: fix this (?)
	//	return sourceValue
	//}

	result := reflect.MakeMap(sourceType)
	mapValueKind := mapValueType.Kind()
	mapRange := sourceValue.MapRange()

	for mapRange.Next() {
		key := mapRange.Key()
		sourceValue := mapRange.Value()

		var destValue reflect.Value
		if p, isNode := sourceValue.Interface().(Node[T]); isNode {
			// This Node MUST be Bindable, otherwise it panics
			destValue = reflect.ValueOf(p.(NodeBinder[T]).Bind(r.g))
		} else if mapValueKind == reflect.Struct || mapValueKind == reflect.Ptr {
			destValue = reflect.ValueOf(r.bindStruct(sourceValue.Interface()))
		} else {
			panic(`try binding unexpected type`)
		}
		result.SetMapIndex(key, destValue)
	}

	return result
}

func isNil(a interface{}) bool {
	return a == nil || (reflect.ValueOf(a).Kind() == reflect.Ptr && reflect.ValueOf(a).IsNil())
}
