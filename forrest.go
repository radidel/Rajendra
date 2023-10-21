package main

import (
	"fmt"
	"github.com/fatih/color"
	"time"
)

var forrestFrames = []string{
	// ... (your ASCII frames here)
}

func main() {
	green := color.New(color.FgGreen)
	red := color.New(color.FgRed)
	blue := color.New(color.FgBlue)
	white := color.New(color.FgWhite)

	colors := []*color.Color{green, red, blue, white}

	for {
		for i, frame := range forrestFrames {
			color := colors[i%len(colors)]
			color.Printf(frame)
			time.Sleep(500 * time.Millisecond) // Adjust the delay as needed
			fmt.Print("\033[H\033[2J") // Clear the screen
		}
	}
}
