# Digital Rotoscope #

This is the repository for the Matlab to C++ conversion of Digital Rotoscope.

### Things that need to be done ###

* Upload new iterations of code
* Finish new iterations of code

## Instructions ##

### How to  compile ###

* Need to build OpenCV on system
* If OpenCV is in custom directory, specify in cmake file
* Compile using cmake and g++ :
```
user:~/rotoscope$ cmake .
user:~/rotoscope$ make
```

### How to run ###

Current version can be called 3 ways:

1. Process single frame of video at given time (assumes frame 0 is background)  
`./program <video> <time>`
2. Process video sequence from given time range (assumes frame 0 is background)  
`./program <video> <start_time> <end_time>`
3. Process video sequence from given time range using specified frame as background  
`./program <video> <start_time> <end_time> <background_time>`

Examples are given in main.cpp. For most usage, the second call type should be used.

## This is the private version ##

* Please don't share the prototype version
* We will make a public version once this is near completion
