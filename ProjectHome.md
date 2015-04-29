This is my CUDA-based implementation of distance computation for two 3D point clouds. It can also compute the distance from a point cloud to a triangle mesh. The documents will be maintained [here](https://sites.google.com/site/wikiforrecheliusprojects/gpudist), not the wiki.

The base line algorithm is very simple: It just computes the distance between each pair of points. Surprisingly, I cannot find related GPU code (although there are multiple papers on GPU-based distance computation). Besides, it actually took me a while to tune the performance, and thus I guess that releasing it can be beneficial to others.

PS. I implemented this library in 2012 spring. Today (2013/08/13) I found two related packages on Google Code.

  * [psa - point set analysis](https://code.google.com/p/psa/) by Thomas Schlömer
  * [gpu-hausdorff](https://code.google.com/p/gpu-hausdorff/)

The second one seem very relevant. I will check which algorithm it implements. One good thing of my package is that it is configured by CMake and thus should be easier to port to other platforms (I tested it on Linux before).
