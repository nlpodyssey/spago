// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package graphviz

import (
	"fmt"

	"github.com/awalterschulze/gographviz"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var timeStepColors = []string{
	"#000000",
	"#5899DA",
	"#E8743B",
	"#19A979",
	"#ED4A7B",
	"#945ECF",
	"#13A4B4",
	"#525DF4",
	"#BF399E",
	"#6C8893",
	"#EE6868",
	"#2F6497",
}

type builder[T mat.DType] struct {
	g  *ag.Graph[T]
	gv gographviz.Interface

	// coloredTimeSteps indicates whether to use different colors for
	// representing nodes with different time-step values.
	coloredTimeSteps bool
	// showNodesWithoutEdges indicates whether to show graph nodes
	// which have no connections.
	showNodesWithoutEdges bool
}

// Option allows to tweak the graphviz generation.
type Option[T mat.DType] func(*builder[T])

// WithColoredTimeSteps enables to use different colors for
// representing nodes with different time-step values.
func WithColoredTimeSteps[T mat.DType](value bool) Option[T] {
	return func(m *builder[T]) {
		m.coloredTimeSteps = value
	}
}

// ShowNodesWithoutEdges enables whether to show graph nodes
// which have no connections.
func ShowNodesWithoutEdges[T mat.DType](value bool) Option[T] {
	return func(m *builder[T]) {
		m.showNodesWithoutEdges = value
	}
}

func newBuilder[T mat.DType](g *ag.Graph[T], options ...Option[T]) *builder[T] {
	b := &builder[T]{
		g:  g,
		gv: gographviz.NewEscape(),
	}
	for _, option := range options {
		option(b)
	}
	return b
}

func (b *builder[T]) build() (gographviz.Interface, error) {
	if err := b.gv.SetDir(true); err != nil {
		return nil, err
	}
	if err := b.gv.AddAttr("", "rankdir", "LR"); err != nil {
		return nil, err
	}

	var nodesWithoutEdges intSet
	if !b.showNodesWithoutEdges {
		nodesWithoutEdges = b.findNodesWithoutEdges()
	}

	lastTimeStep := -1
	for _, node := range b.g.Nodes() {
		if nodesWithoutEdges != nil && nodesWithoutEdges.Has(node.ID()) {
			continue
		}
		if ts := node.TimeStep(); ts != lastTimeStep {
			if err := b.addTimeStepSubGraph(ts); err != nil {
				return nil, err
			}
			lastTimeStep = ts
		}

		switch nt := node.(type) {
		case *ag.Variable[T]:
			if err := b.addVariable(nt); err != nil {
				return nil, err
			}
		case *ag.Operator[T]:
			if err := b.addOperator(nt); err != nil {
				return nil, err
			}
		case *ag.Wrapper[T]:
			if err := b.addWrapper(nt); err != nil {
				return nil, err
			}
		default:
			return nil, fmt.Errorf("unexpected node type %T", node)
		}
	}
	return b.gv, nil
}

func (b *builder[T]) addVariable(v *ag.Variable[T]) error {
	id := fmt.Sprintf("%d", v.ID())
	name := v.Name()
	if name == "" {
		name = "-"
	}
	label := fmt.Sprintf(
		`<
			<FONT COLOR="#707070" POINT-SIZE="11">%d</FONT><BR />
			variable <B>%s</B><BR />
			%s
		>`,
		v.ID(),
		name,
		matrixShapeString(v.Value()),
	)
	attrs := map[string]string{
		"label": label,
		"shape": "box",
		"color": b.timeStepColor(v.TimeStep()),
	}
	parentGraph := b.timeStepGraphName(v.TimeStep())
	return b.gv.AddNode(parentGraph, id, attrs)
}

func (b *builder[T]) addWrapper(v *ag.Wrapper[T]) error {
	if param, ok := v.GradValue.(nn.Param[T]); ok {
		return b.addParam(v, param.Name())
	}

	id := fmt.Sprintf("%d", v.ID())
	label := fmt.Sprintf(
		`<
			<FONT COLOR="#707070" POINT-SIZE="11">%d</FONT><BR />
			wrapper<BR />
			%s
		>`,
		v.ID(),
		matrixShapeString(v.Value()),
	)
	attrs := map[string]string{
		"label": label,
		"shape": "box",
		"color": b.timeStepColor(v.TimeStep()),
	}
	parentGraph := b.timeStepGraphName(v.TimeStep())
	return b.gv.AddNode(parentGraph, id, attrs)
}

func (b *builder[T]) addParam(v *ag.Wrapper[T], name string) error {
	name1 := name
	if name1 == "" {
		name1 = "-"
	}
	id := fmt.Sprintf("%d", v.ID())
	label := fmt.Sprintf(
		`<
			<FONT COLOR="#707070" POINT-SIZE="11">%d</FONT><BR />
			param <B>%s</B><BR />
			%s
		>`,
		v.ID(),
		name1,
		matrixShapeString(v.Value()),
	)
	attrs := map[string]string{
		"label": label,
		"shape": "box",
		"color": b.timeStepColor(v.TimeStep()),
	}
	parentGraph := b.timeStepGraphName(v.TimeStep())
	return b.gv.AddNode(parentGraph, id, attrs)
}

func (b *builder[T]) addOperator(op *ag.Operator[T]) error {
	operatorID := fmt.Sprintf("%d", op.ID())

	label := fmt.Sprintf(
		`<
			<FONT COLOR="#707070" POINT-SIZE="11">%d</FONT><BR />
			<B>%s</B><BR />
			%s
		>`,
		op.ID(),
		op.Name(),
		matrixShapeString(op.Value()),
	)
	attrs := map[string]string{
		"label": label,
		"color": b.timeStepColor(op.TimeStep()),
	}
	parentGraph := b.timeStepGraphName(op.TimeStep())
	if err := b.gv.AddNode(parentGraph, operatorID, attrs); err != nil {
		return err
	}

	for _, operand := range op.Operands() {
		operandID := fmt.Sprintf("%d", operand.ID())
		if err := b.gv.AddEdge(operandID, operatorID, true, nil); err != nil {
			return err
		}
	}
	return nil
}

func (b *builder[T]) addTimeStepSubGraph(timeStep int) error {
	attrs := map[string]string{
		"label":     fmt.Sprintf("Time Step %d", timeStep),
		"color":     b.timeStepColor(timeStep),
		"fontcolor": b.timeStepColor(timeStep),
		"fontsize":  "11",
	}
	return b.gv.AddSubGraph("", b.timeStepGraphName(timeStep), attrs)
}

func (b *builder[T]) timeStepColor(timeStep int) string {
	if !b.coloredTimeSteps {
		return "#000000"
	}
	return timeStepColors[timeStep%len(timeStepColors)]
}

func (b *builder[T]) timeStepGraphName(timeStep int) string {
	if b.g.TimeStep() == 0 {
		return fmt.Sprintf("time_step_%d", timeStep)
	}
	return fmt.Sprintf("cluster_time_step_%d", timeStep)
}

func (b *builder[T]) findNodesWithoutEdges() intSet {
	ids := newIntSet()
	for _, node := range b.g.Nodes() {
		operator, isOperator := node.(*ag.Operator[T])
		if !isOperator || len(operator.Operands()) == 0 {
			ids.Add(node.ID())
			continue
		}
		for _, operand := range operator.Operands() {
			ids.Delete(operand.ID())
		}
	}
	return ids
}

func matrixShapeString[T mat.DType](m mat.Matrix[T]) string {
	if m == nil {
		return ""
	}
	if mat.IsScalar(m) {
		return "scalar"
	}
	return fmt.Sprintf("%d × %d", m.Rows(), m.Columns())
}
