package write

import (
	"fmt"
	"io"
	"os"
	"strings"
	"time"
)

// TODO: Add a result

type WriteSettings struct {
	DisplayWriters []Writer // Where should the display be written. This can be set to nil to avoid all display
}

func DefaultWriteSettings() *WriteSettings {
	return &WriteSettings{
		DisplayWriters: []Writer{{os.Stdout, Displayer}},
	}
}

type Type int

const (
	// Logger is a writer intended to save details of the optimization run
	// for future postprocessing. The data is saved as a csv and data is printed
	// every major interation of the optimizer
	Logger = iota

	// Displayer is a writer intended for human monitoring of the optimization
	// Writes only happen periodically, and an effort is made to align columns
	Displayer
)

type Writer struct {
	io.Writer
	T Type
}

type Value struct {
	Value   interface{}
	Heading string
}

type DataAdder interface {
	AppendWriteData([]*Value) []*Value
}

func writeOptimizationHeader(w io.Writer) {
	w.Write([]byte("Beginning Optimization\n"))
	w.Write([]byte("\n"))
}

const headingInterval = 30
const valueInterval time.Duration = 500 * time.Millisecond

// Display displays stuff. If it's stdio then it only prints at
// specific times. If it's anything else it logs at every iteration
// Assumption is that headings don't change
type Display struct {
	displayValues []*Value

	headings []string
	//headingLengths []int
	values []string
	//valueLengths   []int

	maxLengths []int

	lastHeadingDisplay int
	lastValueDisplay   time.Time

	writerHeadingsWritten bool

	existsDisplayer bool
	existsLogger    bool

	writers []Writer

	dataAdders []DataAdder
}

// accumulateValues gets all of the values from the data adder and stores
// them in display
func (d *Display) accumulateValues() {
	d.displayValues = d.displayValues[:0]
	for _, add := range d.dataAdders {
		d.displayValues = add.AppendWriteData(d.displayValues)
	}
}

func NewDisplay() *Display {
	// return settings so that headings and values are displayed on first iteration
	return &Display{
		lastHeadingDisplay: headingInterval + 1,
		lastValueDisplay:   time.Now().Add(-valueInterval),
	}
}

// AddDataAdder adds a DataAdder to the list of values to be printed/logged.
// This should only be called during initialization
func (d *Display) AddDataAdder(dataAdders ...DataAdder) {
	d.dataAdders = append(d.dataAdders, dataAdders...)
}

// Init initializes the displays for the writers according to their Type
func (d *Display) Init(w *WriteSettings) error {

	d.writers = w.DisplayWriters

	if len(d.writers) == 0 {
		return nil
	}
	d.accumulateValues()

	// get all of the headings
	d.headings = d.headings[:0]
	for _, dat := range d.displayValues {
		d.headings = append(d.headings, dat.Heading)
	}

	// Write the initial headers to all of the writers
	for _, w := range d.writers {
		writeOptimizationHeader(w)
		switch w.T {
		default:
			panic("display: unknown writer type")
		case Logger:
			d.existsLogger = true
			err := d.initLog(w, d.headings)
			if err != nil {
				return err
			}
		case Displayer:
			d.existsDisplayer = true
		}
	}
	return nil
}

// Iterate is the write action performed by display at every iteration
// of the algorithm, as set by the values in the Writers and dataAdders which
// were set during initialization
func (d *Display) Iterate() error {

	var displayValues bool
	var displayHeadings bool

	if d.existsDisplayer {
		// Check if the values need to be displayed
		displayValues = d.shouldDisplayValues()
		if displayValues {
			d.lastValueDisplay = time.Now()
			d.lastHeadingDisplay++
		}

		displayHeadings = d.shouldDisplayHeadings()
		if displayHeadings {
			d.lastHeadingDisplay = 0
		}
	}

	// only accumulate values if needed
	if d.existsLogger || displayValues || displayHeadings {
		d.accumulateValues()
		d.values = d.values[:0]
		for _, v := range d.displayValues {
			str := valueToString(v.Value)
			d.values = append(d.values, str)
		}
	}

	// Find the max length of heading and value
	if displayValues || displayHeadings {
		d.maxLengths = d.maxLengths[:0]
		for i, v := range d.values {
			d.maxLengths = append(d.maxLengths, len(v))
			if len(d.headings[i]) > len(v) {
				d.maxLengths[i] = len(d.headings[i])
			}
		}
	}
	for _, w := range d.writers {
		switch w.T {
		default:
			panic("display: unknown writer type")
		case Logger:
			err := log(w, d.values)
			if err != nil {
				return err
			}
		case Displayer:
			if displayHeadings {
				_, err := w.Write([]byte("\n"))
				if err != nil {
					return err
				}
				err = writeAlignedStrings(w, d.headings, d.maxLengths)
				if err != nil {
					return err
				}
			}
			if displayValues {
				err := writeAlignedStrings(w, d.values, d.maxLengths)
				if err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func (d *Display) shouldDisplayValues() bool {
	// Display values when enough time has elapsed since the last
	// display. This is to limit printing with really quick objective
	// functions
	return time.Since(d.lastValueDisplay) > valueInterval
}

func (d *Display) shouldDisplayHeadings() bool {
	// Display headings again after a certain number of value printings
	return d.lastHeadingDisplay > headingInterval
}

// Init writes the headings to the io.Writer
func (d *Display) initLog(w io.Writer, headings []string) error {
	// Write all of the data headings in csv format
	for i, heading := range headings {
		_, err := w.Write([]byte(heading))
		if err != nil {
			return err
		}
		if i != len(headings) {
			_, err := w.Write([]byte(","))
			if err != nil {
				return err
			}
		}
	}
	_, err := w.Write([]byte("\n"))
	if err != nil {
		return err
	}
	return nil
}

func writeAlignedStrings(w io.Writer, strs []string, maxLengths []int) error {
	for i, str := range strs {
		s := str + strings.Repeat(" ", maxLengths[i]-len(str)) + "\t"
		_, err := w.Write([]byte(s))
		if err != nil {
			return err
		}
	}
	_, err := w.Write([]byte("\n"))
	return err
}

// Log adds the data to the io.Writer
func log(w io.Writer, values []string) error {
	for i, value := range values {
		_, err := w.Write([]byte(value))
		if err != nil {
			return err
		}
		if i != len(values) {
			_, err := w.Write([]byte(","))
			if err != nil {
				return err
			}
		}
	}
	_, err := w.Write([]byte("\n"))
	if err != nil {
		return err
	}
	return nil
}

func valueToString(v interface{}) string {
	switch v.(type) {
	case int:
		return fmt.Sprintf("%d", v)
	case float64:
		return fmt.Sprintf("%e", v)
	case string:
		return fmt.Sprintf("%s", v)
	default:
		return fmt.Sprintf("%v", v)
	}
}
