// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"reflect"
)

type reifier[T mat.DType] struct {
	g    *ag.Graph[T]
	mode ProcessingMode
}

func newReifier[T mat.DType](g *ag.Graph[T], mode ProcessingMode) *reifier[T] {
	return &reifier[T]{
		g:    g,
		mode: mode,
	}
}

func (r *reifier[T]) reify(m Model[T]) Model[T] {
	p := r.reifyStruct(m).(Model[T])
	p.InitProcessor()
	return p
}

func (r *reifier[_]) reifyStruct(rawSource interface{}) interface{} {
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

		if tag.Scope == processorModuleFieldScope || !sourceField.CanInterface() {
			continue // skip any initialization
		}

		if tag.Scope == modelModuleFieldScope {
			destField.Set(sourceField)
			continue
		}

		r.reifyStructField(sourceField, destField, tag)
	}

	if !sourceIsPointer {
		return dest.Interface()
	}
	return destPointer.Interface()
}

func (r *reifier[T]) reifyStructField(sourceField, destField reflect.Value, tag moduleFieldTag) {
	switch sourceFieldT := sourceField.Interface().(type) {
	case *ag.Graph[T]:
		destField.Set(reflect.ValueOf(r.g))
	case ProcessingMode:
		destField.Set(reflect.ValueOf(r.mode))
	case BaseModel[T], *BaseModel[T]:
		destField.Set(reflect.ValueOf(r.reifyStruct(sourceFieldT)))
	case Param[T]:
		destField.Set(reflect.ValueOf(r.reifyParam(sourceFieldT.(*param[T]))))
	case []Param[T]:
		destField.Set(reflect.ValueOf(r.reifyParamSlice(sourceFieldT)))
	case Model[T]:
		destField.Set(reflect.ValueOf(r.reifyModel(sourceFieldT)))
	case []Model[T]:
		destField.Set(reflect.ValueOf(r.reifyModelSlice(sourceFieldT)))
	default:
		switch sourceField.Kind() {
		case reflect.Slice:
			destField.Set(r.reifySlice(sourceField, tag))
		case reflect.Map:
			destField.Set(r.reifyMap(sourceField, tag))
		case reflect.Struct, reflect.Ptr:
			if sourceField.Kind() == reflect.Ptr && sourceField.IsNil() {
				return
			}
			if tag.Type == paramsModuleFieldType {
				destField.Set(reflect.ValueOf(r.reifyStruct(sourceFieldT)))
			} else {
				destField.Set(sourceField)
			}
		default:
			destField.Set(sourceField)
		}
	}
}

func (r *reifier[T]) reifyModel(sourceField Model[T]) Model[T] {
	if isNil(sourceField) {
		return sourceField
	}
	p := Reify(sourceField, r.g, r.mode)
	p.InitProcessor()
	return p
}

func (r *reifier[T]) reifyModelSlice(sourceField []Model[T]) []Model[T] {
	result := make([]Model[T], len(sourceField))
	for i := 0; i < len(sourceField); i++ {
		result[i] = r.reifyModel(sourceField[i])
	}
	return result
}

func (r *reifier[T]) reifyParam(sourceField *param[T]) Param[T] {
	return sourceField.wrappedParam(r.g)
}

func (r *reifier[T]) reifyParamSlice(sourceField []Param[T]) []Param[T] {
	result := make([]Param[T], len(sourceField))
	for i := 0; i < len(sourceField); i++ {
		result[i] = r.reifyParam(sourceField[i].(*param[T]))
	}
	return result
}

func (r *reifier[T]) reifySlice(sourceField reflect.Value, tag moduleFieldTag) reflect.Value {
	length := sourceField.Len()
	result := reflect.MakeSlice(sourceField.Type(), length, length)
	isParamsTag := tag.Type == paramsModuleFieldType

	for i := 0; i < length; i++ {
		sourceItem := sourceField.Index(i)
		switch sourceItem.Kind() {
		case reflect.Struct, reflect.Ptr, reflect.Interface:
			_, isModule := sourceItem.Interface().(Model[T])
			if isParamsTag || isModule {
				result.Index(i).Set(reflect.ValueOf(r.reifyStruct(sourceItem.Interface())))
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

// TODO: check if this works well with generics!
var paramInterfaceName = reflect.TypeOf((*Param[float32])(nil)).Elem().Name()

func (r *reifier[T]) reifyMap(sourceValue reflect.Value, tag moduleFieldTag) reflect.Value {
	sourceType := reflect.TypeOf(sourceValue.Interface())
	mapValueType := sourceType.Elem()

	if mapValueType.Name() != paramInterfaceName && tag.Type != paramsModuleFieldType {
		return sourceValue
	}

	result := reflect.MakeMap(sourceType)
	mapValueKind := mapValueType.Kind()
	mapRange := sourceValue.MapRange()

	for mapRange.Next() {
		key := mapRange.Key()
		sourceValue := mapRange.Value()

		var destValue reflect.Value
		if p, isParam := sourceValue.Interface().(*param[T]); isParam {
			destValue = reflect.ValueOf(r.reifyParam(p))
		} else if mapValueKind == reflect.Struct || mapValueKind == reflect.Ptr {
			destValue = reflect.ValueOf(r.reifyStruct(sourceValue.Interface()))
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
