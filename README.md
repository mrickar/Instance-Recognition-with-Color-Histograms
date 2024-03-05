# Instance-Recognition-with-Color-Histograms

This project is the submission for the Take Home Exam given in CENG483 Introduction to Computer Vision at METU. The explanation of the homework is included in the the1_task.pdf file.

## Table of Contents
- [Instance-Recognition-with-Color-Histograms](#instance-recognition-with-color-histograms)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [How to use](#how-to-use)

## Project Overview
The purpose of this take-home exam is to implement a simple instance recognition with color histograms. The Instance Recognition system based on different types
of color histograms and evaluate it. The observations and evalutaions are reported in report.pdf file.

## How to use

There are 4 main functions.


```python
convert_queries_to_hsv()
"""
Should be called before the below function if HSV values will be used.
""" 
```

```python
threeD_run(interval,rgb_hsv)
"""
this function should be called if 3D color histograms are going to be used and there will not be any grids.
Args:
    interval (int): interval size of histograms. 
    rgb_hsv (str): this decides wheter should use RGB or HSV values. Predefined RGB or HSV variable should be passed as parameter.
"""
```

```python
per_channel_run(interval,rgb_hsv)
"""
this function should be called if per-channel color histograms are going to be used and there will not be any grids.
Args:
    interval (int): interval size of histograms. 
    rgb_hsv (str): this decides wheter should use RGB or HSV values.Predefined RGB or HSV variable should be passed as parameter.
"""
```

```python
grid_based_run(interval,rgb_hsv,per_channel_3d,grid_size)
"""
this function should be called if Grid Based histograms are going to be used.
Args:
    interval (int): interval size at histograms.
    rgb_hsv (str): this decides wheter should use RGB or HSV values.
    per_channel_3d (str): this decides wheter should use per-channel or 3D color histogram. Predefined PER_CHANNEL or THREE_D variable should be passed as parameter.
    grid_size (int): this decides the grid size. If 2x2 spatial grids wanted grid_size should be passed as 2.
"""
```
