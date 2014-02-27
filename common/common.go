package common

import (
	"fmt"
	"math"
	"time"

	"github.com/btracey/opt/write"
)

type Initer interface {
	Init()
}

type Resulter interface {
	Result()
}

// Helper routines for wrapping the objective function
//
// If the function is an initer it will be called once
// If the function is a statuse
type ObjectiveWrapper struct {
	fun        interface{}
	initCalled bool
}

func (o *ObjectiveWrapper) Init(objectiveFunction interface{}) {

	if o.initCalled {
		return
	}
	o.initCalled = true
	o.fun = objectiveFunction

	initer, ok := objectiveFunction.(Initer)
	if ok {
		initer.Init()
	}
}

func (o *ObjectiveWrapper) Status() Status {
	statuser, isStatuser := o.fun.(Statuser)
	if isStatuser {
		return statuser.Status()
	}
	return Continue
}

func (o *ObjectiveWrapper) Result() {
	resulter, ok := o.fun.(Resulter)
	if ok {
		resulter.Result()
	}
}

func (o *ObjectiveWrapper) AppendWriteData(v []*write.Value) []*write.Value {
	dataWriter, ok := o.fun.(write.DataAdder)
	if ok {
		return dataWriter.AppendWriteData(v)
	}
	return v
}

type SingleOutputSettings struct {
	GradAbsTol    float64 // Relative tolerance of the gradient norm
	GradRelTol    float64 // Absolute tolerance of the gradient norm
	GradRelWindow int     // Window for measuring the relative value
	ObjAbsTol     float64 // Relative tolerance of the objective function value
	ObjRelTol     float64 // Relative tolerance of the objective function norm
	ObjRelWindow  int     // Window for measuring the relative value
}

func DefaultSingleOutputSettings() *SingleOutputSettings {
	return &SingleOutputSettings{
		GradAbsTol:    1e-6,
		GradRelTol:    -1,
		GradRelWindow: 5,
		ObjAbsTol:     math.NaN(),
		ObjRelTol:     -1,
		ObjRelWindow:  5,
	}
}

// Helper struct for optimizers who have a single output
type SingleOutput struct {
	grad *UniToler
	obj  *UniToler
}

func NewSingleOutput() *SingleOutput {
	return &SingleOutput{
		grad: &UniToler{},
		obj:  &UniToler{},
	}
}

func (s *SingleOutput) Init(settings *SingleOutputSettings, initObj, initGrad float64) {
	s.grad.Init(settings.GradAbsTol, settings.GradRelTol, settings.GradRelWindow, initGrad)
	s.obj.Init(settings.ObjAbsTol, settings.ObjRelTol, settings.ObjRelWindow, initObj)
}

func (s *SingleOutput) Iterate(gradNrm float64, fun float64) {
	s.grad.Add(gradNrm)
	s.obj.Add(fun)
}

func (s *SingleOutput) Status() Status {
	if s.grad.AbsConverged() {
		fmt.Println("grad converged")
		return GradAbsTol
	}
	if s.grad.RelConverged() {
		return GradChangeTol
	}
	if s.obj.AbsConverged() {
		return ObjAbsTol
	}
	if s.obj.RelConverged() {
		return ObjChangeTol
	}
	return Continue
}

// CommonSettings is a set of options available to all optimizers
type CommonSettings struct {
	MaximumIterations          int           // Sets the maximum number of major iterations that can occur
	MaximumFunctionEvaluations int           // Sets the maximum number of function evaluations that can occur
	MaximumRuntime             time.Duration // Sets the maximum runtime that can elapse
	*write.WriteSettings
}

// DefaultSettings returns the default settings for the common structure
func DefaultCommonSettings() *CommonSettings {
	return &CommonSettings{
		MaximumIterations:          -1, // Defaults to no maximum iterations
		MaximumFunctionEvaluations: -1, // Defaults to no maximum function evaluations
		MaximumRuntime:             -1, // Defaults to no maximum runtime
		WriteSettings:              write.DefaultWriteSettings(),
	}
}

// CommonResult is a list of results from the common structure
type CommonResult struct {
	Iterations          int           // Total number of iterations taken by the optimizer
	FunctionEvaluations int           // Total number of function evaluations taken by the optimizer
	Runtime             time.Duration // Total runtime elapsed during the optimization
	Status              Status        // How did the optimizer end
}

// Common provides routines for controlling the settings provided by common.
type Common struct {
	iter      int
	funEvals  int
	startTime time.Time

	settings *CommonSettings

	*write.Display
	*ObjectiveWrapper
}

// NewOptCommon creates a new OptCommon structure, and adds itself to the datawriter
func NewCommon() *Common {
	c := &Common{
		Display:          write.NewDisplay(),
		ObjectiveWrapper: &ObjectiveWrapper{},
	}
	c.AddDataAdder(c, c.ObjectiveWrapper)
	return c
}

// Init initializes all of the values in common at the start of the optimization
func (c *Common) Init(settings *CommonSettings, objectiveFunction interface{}) {
	c.iter = 0
	c.funEvals = 0
	c.startTime = time.Now()

	c.settings = settings

	// Initialize the display, adding common as one of the dataAdders
	c.Display.Init(c.settings.WriteSettings)
	c.ObjectiveWrapper.Init(objectiveFunction)
}

// AddToDisplay adds the components of common to the display structure
func (c *Common) AppendWriteData(d []*write.Value) []*write.Value {
	d = append(d, &write.Value{Heading: "Iter", Value: c.iter})
	d = append(d, &write.Value{Heading: "FnEval", Value: c.funEvals})
	return d
}

// Note: These have names that are different because we want optimizers
// to specifically implement all of them. If it has the name Status(), then
// an optimizer will implement by embedding common

// Status checks if any of the data controlled by common has converged (runtime, funevals, etc.)
func (c *Common) Status() Status {

	status := c.ObjectiveWrapper.Status()
	if status != Continue {
		return status
	}

	if c.settings.MaximumIterations > -1 && c.iter > c.settings.MaximumIterations {
		return MaximumIterations
	}
	if c.settings.MaximumFunctionEvaluations > -1 && c.funEvals > c.settings.MaximumFunctionEvaluations {
		return MaximumFunctionEvaluations
	}
	if c.settings.MaximumRuntime > -1 && time.Since(c.startTime) > c.settings.MaximumRuntime {
		return MaximumRuntime
	}
	return Continue
}

// CommonResult returns the results from the common structure
func (c *Common) Result(status Status) *CommonResult {
	c.ObjectiveWrapper.Result()
	r := &CommonResult{
		Iterations:          c.iter,
		FunctionEvaluations: c.funEvals,
		Runtime:             time.Since(c.startTime),
		Status:              status,
	}
	return r
}

// Iterate performs an iteration of the common structure, incrementing
// the iteration, appending the number of function evaluations, and
// writing to the writers
func (c *Common) Iterate(nFunEvals int) {
	c.iter++
	c.funEvals += nFunEvals
	c.Display.Iterate()
}
