// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package graphviz

import (
	"fmt"
	"github.com/awalterschulze/gographviz"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

type builder struct {
	g   *ag.Graph
	gv  gographviz.Interface
	opt Options
}

func newBuilder(g *ag.Graph, options Options) *builder {
	return &builder{
		g:   g,
		gv:  gographviz.NewEscape(),
		opt: options,
	}
}

func (b *builder) build() (gographviz.Interface, error) {
	if err := b.gv.SetDir(true); err != nil {
		return nil, err
	}
	if err := b.gv.AddAttr("", "rankdir", "LR"); err != nil {
		return nil, err
	}

	lastTimeStep := -1
	for _, node := range b.g.Nodes() {
		if ts := node.TimeStep(); ts != lastTimeStep {
			if err := b.addTimeStepSubGraph(ts); err != nil {
				return nil, err
			}
			lastTimeStep = ts
		}

		switch nt := node.(type) {
		case *ag.Variable:
			if err := b.addVariable(nt); err != nil {
				return nil, err
			}
		case *ag.Operator:
			if err := b.addOperator(nt); err != nil {
				return nil, err
			}
		case *ag.Wrapper:
			if param, ok := nt.GradValue.(nn.Param); ok {
				if err := b.addParam(nt, param.Name()); err != nil {
					return nil, err
				}
				continue
			}
			if err := b.addWrapper(nt); err != nil {
				return nil, err
			}
		default:
			return nil, fmt.Errorf("unexpected node type %T", node)
		}
	}
	return b.gv, nil
}

func (b *builder) addVariable(v *ag.Variable) error {
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

func (b *builder) addWrapper(v *ag.Wrapper) error {
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

func (b *builder) addParam(v *ag.Wrapper, name string) error {
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

func (b *builder) addOperator(op *ag.Operator) error {
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

func (b *builder) addTimeStepSubGraph(timeStep int) error {
	attrs := map[string]string{
		"label":     fmt.Sprintf("Time Step %d", timeStep),
		"color":     b.timeStepColor(timeStep),
		"fontcolor": b.timeStepColor(timeStep),
		"fontsize":  "11",
	}
	return b.gv.AddSubGraph("", b.timeStepGraphName(timeStep), attrs)
}

func (b *builder) timeStepColor(timeStep int) string {
	if !b.opt.ColoredTimeSteps {
		return "#000000"
	}
	return timeStepColors[timeStep%len(timeStepColors)]
}

func (b *builder) timeStepGraphName(timeStep int) string {
	if b.g.TimeStep() == 0 {
		return fmt.Sprintf("time_step_%d", timeStep)
	}
	return fmt.Sprintf("cluster_time_step_%d", timeStep)
}

func matrixShapeString(m mat.Matrix) string {
	if m == nil {
		return ""
	}
	if m.IsScalar() {
		return "scalar"
	}
	return fmt.Sprintf("%d × %d", m.Rows(), m.Columns())
}
