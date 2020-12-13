// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utils

import "reflect"

func ReverseInPlace(s interface{}) {
	n := reflect.ValueOf(s).Len()
	swap := reflect.Swapper(s)
	for i, j := 0, n-1; i < j; i, j = i+1, j-1 {
		swap(i, j)
	}
}

func ForEachField(i interface{}, callback func(field interface{}, name string, tag reflect.StructTag)) {
	v := reflect.ValueOf(i).Elem()
	t := reflect.TypeOf(i).Elem()
	length := v.NumField()
	for i := 0; i < length; i++ {
		vField := v.Field(i)
		tField := t.Field(i)
		if vField.CanInterface() {
			if vField.Kind() == reflect.Ptr && vField.IsNil() {
				continue
			}
			callback(vField.Interface(), tField.Name, tField.Tag)
		}
	}
}

func TypeName(instance interface{}) string {
	t := reflect.TypeOf(instance)
	if t.Kind() == reflect.Ptr {
		return t.Elem().Name()
	}
	return t.Name()
}

func Name(i interface{}) string {
	if IsStruct(i) {
		return reflect.TypeOf(i).String()
	}
	//pointer
	return reflect.TypeOf(i).Elem().String()
}

func IsStruct(i interface{}) bool { return reflect.ValueOf(i).Kind() == reflect.Struct }
