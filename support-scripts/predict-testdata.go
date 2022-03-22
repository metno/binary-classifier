package main

// Run predictions concurrently.
// Usage CUDA_VISIBLE_DEVICES="" go run support-scripts/predict-testdata.go  -labelfile /home/espenm/data/scenes/test_data.txt --modelpath checkpoints/saved_model_009.pb --datadir /home/espenm/data/scenes
import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
)

type labelPath struct {
	Path       string
	Label      int
	Prediction int
}

func parseInt(value string) int {
	if len(value) == 0 {
		return 0
	}
	i, err := strconv.Atoi(value)
	if err != nil {
		return -2
	}
	return i
}

// /home/espenm/space/projects/cc-classifier/predict.py --modeldir /home/espenm/space/projects/models/v24_9999 --epoch 831 --filename /lustre/storeB/project/metproduction/products/webcams/2018/06/14/13/13_20180614T2200Z.jpg  2>/dev/null
func execCommand(command string, arg ...string) (string, string, error) {
	cmd := exec.Command(command, arg...)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	outStr, errStr := stdout.String(), stderr.String()
	return outStr, errStr, err

}

func readLabels(path string, dataDir string) ([]labelPath, error) {
	items := []labelPath{}

	r := regexp.MustCompile(`(\S+) (\d)`)

	file, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		matches := r.FindStringSubmatch(line)
		if len(matches) != 3 {
			fmt.Printf("Could not parse input %s\n", line)
			return nil, fmt.Errorf("could not parse input %s", line)
		}

		p := matches[1]
		cclabel := parseInt(matches[2])

		path := fmt.Sprintf("%s/%s", dataDir, p)
		item1 := labelPath{Path: path, Label: cclabel}
		items = append(items, item1)

	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return items, nil
}

// Here's the worker, of which we'll run several
// concurrent instances. These workers will receive
// work on the `jobs` channel and send the corresponding
// results on `results`.
func worker(id int, jobs <-chan labelPath, modelpath string, results chan<- labelPath) {
	count := 0
	for j := range jobs {

		count++
		///fmt.Println("worker", id, "started  job", j, "Path:", j.Path)
		//os.Setenv("CUDA_VISIBLE_DEVICES", "")
		outStr, errStr, err := execCommand(os.Getenv("PROJECT_HOME_CNN")+"/predict.py",
			"--modelpath", modelpath,
			"--filename", j.Path)

		if err != nil {
			fmt.Printf("predict-testdata: Failed: %v\n", err)
			fmt.Printf("predict-testdata: Failed: %s\n", errStr)
		}

		j.Prediction = parseInt(strings.TrimRight(outStr, "\n"))
		results <- j
	}
}

func main() {
	if os.Getenv("PROJECT_HOME_CNN") == "" {
		log.Println("Error: environment variable PROJECT_HOME_CNN not set. Exiting")
	}

	labelFile := flag.String("labelfile", "", "Path to the labelsfile")
	modelPath := flag.String("modelpath", "", "Path to the model")
	datadir := flag.String("datadir", "", "Path to the image dir")
	flag.Parse()
	if *labelFile == "" || *modelPath == "" {
		flag.Usage()
		return
	}

	//cpus := runtime.NumCPU()
	cpus := 12
	//runtime.GOMAXPROCS(cpus)

	labelPaths, err := readLabels(*labelFile, *datadir)
	if err != nil {
		log.Fatal(err)
	}

	// In order to use our pool of workers we need to send
	// them work and collect their results. We make 2
	// channels for this.
	jobs := make(chan labelPath, len(labelPaths))
	results := make(chan labelPath, len(labelPaths))

	// This starts up cpus workers, initially blocked
	// because there are no jobs yet.
	for w := 0; w < cpus; w++ {
		go worker(w, jobs, *modelPath, results)
	}

	// Here we send `len(labelPaths)` jobs and then `close` that
	// channel to indicate that's all the work we have.
	for j := 0; j < len(labelPaths); j++ {

		jobs <- labelPaths[j]
	}
	close(jobs)

	// Finally we collect all the results of the work.
	for a := 0; a < len(labelPaths); a++ {
		res := <-results
		fmt.Printf("%s %d %d\n", res.Path, res.Label, res.Prediction)
	}

}
