// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"reflect"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

type binder[T mat.DType] struct {
	g *ag.Graph[T]
}

func (r *binder[T]) bind(m ag.Differentiable[T]) ag.Differentiable[T] {
	return r.bindStruct(m).(ag.Differentiable[T])
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
		tag, err := parseModuleFieldTag(sourceType.Field(fieldIndex).Tag.Get("spago"))
		if err != nil {
			panic(err)
		}
		if !sourceField.CanInterface() {
			continue // skip any initialization
		}
		r.bindStructField(sourceField, destField, tag)
	}

	if !sourceIsPointer {
		return dest.Interface()
	}
	return destPointer.Interface()
}

func (r *binder[T]) bindStructField(sourceField, destField reflect.Value, tag moduleFieldTag) {
	switch sourceFieldT := sourceField.Interface().(type) {
	case *ag.Graph[T]:
		destField.Set(reflect.ValueOf(r.g))
	case Param[T]:
		destField.Set(reflect.ValueOf(r.bindParam(sourceFieldT.(*BaseParam[T]))))
	case ag.Differentiable[T]:
		destField.Set(reflect.ValueOf(r.bindDifferentiable(sourceFieldT)))
	default:
		switch sourceField.Kind() {
		case reflect.Slice:
			destField.Set(r.bindSlice(sourceField, tag))
		case reflect.Map:
			destField.Set(r.bindMap(sourceField, tag))
		case reflect.Struct, reflect.Ptr:
			if sourceField.Kind() == reflect.Ptr && sourceField.IsNil() {
				return
			}
			if tag.Type == paramsModuleFieldType {
				destField.Set(reflect.ValueOf(r.bindStruct(sourceFieldT)))
			} else {
				destField.Set(sourceField)
			}
		default:
			destField.Set(sourceField)
		}
	}
}

func (r *binder[T]) bindDifferentiable(sourceField ag.Differentiable[T]) ag.Differentiable[T] {
	if isNil(sourceField) {
		return sourceField
	}
	return r.bindStruct(sourceField).(ag.Differentiable[T])
}

func (r *binder[T]) bindDifferentiableSlice(sourceField []ag.Differentiable[T]) []ag.Differentiable[T] {
	result := make([]ag.Differentiable[T], len(sourceField))
	for i := 0; i < len(sourceField); i++ {
		if isNil(sourceField) {
			return sourceField
		}
		result[i] = r.bindStruct(sourceField[i]).(ag.Differentiable[T])
	}
	return result
}

func (r *binder[T]) bindSlice(sourceField reflect.Value, tag moduleFieldTag) reflect.Value {
	length := sourceField.Len()
	result := reflect.MakeSlice(sourceField.Type(), length, length)
	isParamsTag := tag.Type == paramsModuleFieldType

	for i := 0; i < length; i++ {
		sourceItem := sourceField.Index(i)
		switch sourceItem.Kind() {
		case reflect.Struct, reflect.Ptr, reflect.Interface:
			_, isDifferentiable := sourceItem.Interface().(ag.Differentiable[T])
			_, isParam := sourceItem.Interface().(Param[T])
			if isParamsTag || isDifferentiable {
				result.Index(i).Set(reflect.ValueOf(r.bindStruct(sourceItem.Interface())))
			} else if isParam {
				result.Index(i).Set(reflect.ValueOf(r.bindParam(sourceItem.Interface().(Param[T]))))
			} else {
				return sourceField
			}
		default:
			if isParamsTag {
				panic(`nn: "params"-tagged slice contains items with unexpected type`)
			}
			return sourceField
		}
	}
	return result
}

func paramInterfaceNamePrefix[T mat.DType]() string {
	return reflect.TypeOf((*Param[T])(nil)).Elem().Name()
}

func (r *binder[T]) bindMap(sourceValue reflect.Value, tag moduleFieldTag) reflect.Value {
	sourceType := reflect.TypeOf(sourceValue.Interface())
	mapValueType := sourceType.Elem()

	if mapValueType.Name() != paramInterfaceNamePrefix[T]() && tag.Type != paramsModuleFieldType {
		return sourceValue
	}

	result := reflect.MakeMap(sourceType)
	mapValueKind := mapValueType.Kind()
	mapRange := sourceValue.MapRange()

	for mapRange.Next() {
		key := mapRange.Key()
		sourceValue := mapRange.Value()

		var destValue reflect.Value
		if p, isParam := sourceValue.Interface().(*BaseParam[T]); isParam {
			destValue = reflect.ValueOf(r.bindParam(p))
		} else if mapValueKind == reflect.Struct || mapValueKind == reflect.Ptr {
			destValue = reflect.ValueOf(r.bindStruct(sourceValue.Interface()))
		} else {
			panic(`nn: "params"-tagged map contains values with unexpected type`)
		}
		result.SetMapIndex(key, destValue)
	}

	return result
}

func isNil(a interface{}) bool {
	return a == nil || (reflect.ValueOf(a).Kind() == reflect.Ptr && reflect.ValueOf(a).IsNil())
}

func (r *binder[T]) bindParam(p Param[T]) Param[T] {
	if _, ok := p.(*paramNode[T]); ok {
		panic("nn: impossible to bind a param node.")
	}
	if p.RequiresGrad() {
		return &paramNode[T]{Param: p, Node: r.g.NewWrap(p)}
	}
	return &paramNode[T]{Param: p, Node: r.g.NewWrapNoGrad(p)}
}

func (r *binder[T]) bindParamSlice(sourceField []Param[T]) []Param[T] {
	result := make([]Param[T], len(sourceField))
	for i := 0; i < len(sourceField); i++ {
		result[i] = r.bindParam(sourceField[i].(*BaseParam[T]))
	}
	return result
}
