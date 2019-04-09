how to write type_dict.csv

0. delimiter is always ; not ,

1. write the header as below:
var_name;type;default;bound;dist

2. write the each columns as followed:
var_name    -> batch_size   (don't include space)
type        -> int          (notice: basically int or float)
default     -> 10
bound       -> [0.0, 1.0]   (notice: [possible minimum, possible maximum])
dist        -> uniform      (notice: choices are only "uniform" or "log{}". )