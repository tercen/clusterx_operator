# clusterx operator

#### Description
`clusterx` operator performs a fast clustering by automatic search and find of density peaks

##### Usage
Input projection|.
---|---
`row`   | represents the variables (e.g. channels, markers)
`col`   | represents the observations (e.g. cells, samples, individuals) 
`y-axis`| is the measurement value


Input parameters|.
---|---
`dimReduction`   | type of reduction to perform, `pca`, `tsne`, `NULL`, default is `NULL`
`outDim`   | number of demensions to return, default 2

Output relations|.
---|---
`cluster`| character, returns a cluster id per column (e.g. per cell)

##### Details
`clusterx` operator performs a fast clustering by automatic search and find of density peaks.


#### References
see  https://github.com/JinmiaoChenLab/ClusterX


##### See Also
[clusterx](https://github.com/tercen/clusterx_operator)

#### Examples

