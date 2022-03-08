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
	// Bind returns a Node interface to create a graph-bound version of the receiver node.
	Bind(g *Graph[T]) Node[T]
}

type graphBinder[T mat.DType] struct {
	session SessionProvider[T]
}

func (r *graphBinder[T]) newBoundStruct(rawSource any) any {
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

func (r *graphBinder[T]) bindStructField(sourceField, destField reflect.Value) {
	switch sourceFieldT := sourceField.Interface().(type) {
	case Node[T]:
		// This Node MUST be Bindable, otherwise it panics
		destField.Set(reflect.ValueOf(sourceFieldT.(NodeBinder[T]).Bind(r.session.Graph())))
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
		case reflect.Interface:
			if sourceField.Type().Name() == reflect.TypeOf((*SessionProvider[T])(nil)).Elem().Name() {
				destField.Set(reflect.ValueOf(r.session))
				return
			}
			destField.Set(sourceField)
		default:
			destField.Set(sourceField)
		}
	}
}

func (r *graphBinder[T]) bindDifferentiable(sourceField Differentiable[T]) Differentiable[T] {
	if isNil(sourceField) {
		return sourceField
	}
	return r.newBoundStruct(sourceField).(Differentiable[T])

}

func (r *graphBinder[T]) bindDifferentiableSlice(sourceField []Differentiable[T]) []Differentiable[T] {
	result := make([]Differentiable[T], len(sourceField))
	for i := 0; i < len(sourceField); i++ {
		if isNil(sourceField) {
			return sourceField
		}
		result[i] = r.newBoundStruct(sourceField[i]).(Differentiable[T])
	}
	return result
}

func (r *graphBinder[T]) bindSlice(sourceField reflect.Value) reflect.Value {
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

func (r *graphBinder[T]) bindMap(sourceValue reflect.Value) reflect.Value {
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
			destValue = reflect.ValueOf(p.(NodeBinder[T]).Bind(r.session.Graph()))
		} else if mapValueKind == reflect.Struct || mapValueKind == reflect.Ptr {
			destValue = reflect.ValueOf(r.newBoundStruct(sourceValue.Interface()))
		} else {
			panic(`try binding unexpected type`)
		}
		result.SetMapIndex(key, destValue)
	}

	return result
}

func isNil(a any) bool {
	return a == nil || (reflect.ValueOf(a).Kind() == reflect.Ptr && reflect.ValueOf(a).IsNil())
}
