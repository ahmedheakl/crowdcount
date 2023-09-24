# CrowdCounting Made Easy 🤓 with  CNN-based Cascaded Multi-task

This is a packaging implementation of the paper [CNN-based Cascaded Multi-task Learning of High-level Prior and Density Estimation for Crowd Counting](https://arxiv.org/pdf/1707.09605.pdf) for single image crowd counting which is accepted at [AVSS 2017](http://www.avss2017.org/)

The package is compatible with all operating systems, provides a staggering fast and accurate prediction. It achieves a min of 20 fps on a 6 core intel cpu.

# Installation 
```bash
pip install ezcrowdcount
```

# Usage

To run inference on your favorite image/video simply run the following on your terminal/console:

```bash
crowdcount --mode video --path /path/to/video
```

```python
"""
mode (str): Whether to run prediction on video or image
path (str | int): Path to video or image. It can be an index to a camera feed, or a URL also. (Default = 0).
"""
```

The inference will run on your GPU (if available), and will be viewed right in front of you 👀
Also, the number of people during each frame will be printed on your console/terminal.

# Demo
**Input Image:**

![Input Image](https://github.com/ahmedheakl/crowdcount/blob/master/imgs/sample.jpg?raw=true)

**Result Image:**

![Result Image](https://github.com/ahmedheakl/crowdcount/blob/master/imgs/sample-result.png?raw=true)

#### Number of people: 165.8 🎉