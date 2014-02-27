package common

type Statuser interface {
	Status() Status
}

// CheckConvergence checks the convergence of a variadic
// number of converges and returns the first non-nil result
func CheckStatus(cs ...Statuser) Status {
	for _, val := range cs {
		c := val.Status()
		if c != Continue {
			return c
		}
	}
	return Continue
}

// NewStatus is used to get a unique value for Status to avoid any accidental
// collisions. NewStatus is not thread-safe as it is intended to only be used
// during initialization
func NewStatus(str string) Status {
	lastStatus++
	statusStrings[lastStatus] = str
	return Status(lastStatus)
}

var statusStrings map[Status]string

func init() {
	statusStrings = make(map[Status]string)
	statusStrings[Continue] = "Continue"
	statusStrings[LocChangeTol] = "LocChangeTol"
	statusStrings[GradAbsTol] = "GradAbsTol"
	statusStrings[ObjAbsTol] = "ObjAbsTol"
	statusStrings[ObjChangeTol] = "ObjChangeTol"
	statusStrings[WolfeConditionsMet] = "WolfeConditionsMet"
	statusStrings[BoundsConverged] = "BoundsConverged"

	statusStrings[UserFunctionError] = "ErrorInUserFunction"
	statusStrings[Infeasible] = "ProblemInfeasible"
	statusStrings[MaximumIterations] = "MaximumIterations"
	statusStrings[MaximumFunctionEvaluations] = "MaximumFunctionEvaluations"
	statusStrings[MaximumRuntime] = "MaximumRuntimeElapsed"
	statusStrings[LinesearchFailure] = "LinesearchFailedToConverge"
}

// Status is a type for expressing if the optimizer has finished or not
// Zero signifies no convergence or error so the optimizer should continue.
// Positive values indicate successful convergence
// negative values express failure for some way
//
// If a custom status value is desired, NewStatus should be called. NewStatus
// is not thread-safe as it is intended to only be used during initialization
type Status int

func (s Status) String() string {
	str, ok := statusStrings[s]
	if !ok {
		return "UnregisteredStatus"
	}
	return str
}

const (
	Continue Status = iota
	LocChangeTol
	GradAbsTol
	GradChangeTol
	ObjAbsTol
	ObjChangeTol
	WolfeConditionsMet
	BoundsConverged
)

const (
	_                        = iota
	UserFunctionError Status = -1 * iota
	OptimizerError
	Infeasible
	MaximumIterations
	MaximumFunctionEvaluations
	MaximumRuntime
	LinesearchFailure
)

var lastStatus Status = 256
