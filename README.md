# Clustering

This crate provides an easy and efficient way to perform kmeans 
clustering on arbitrary data. The algo is initialized with kmeans++
for best performance of the clustering. 

There are three goals to this implementation of the kmeans algorithm:

1. it must be generic
2. it must be easy to use
3. it must be reasonably fast

## Example

```rust
use clustering::*;

let n_samples    = 20_000; // # of samples in the example
let n_dimensions =    200; // # of dimensions in each sample
let k            =      4; // # of clusters in the result
let max_iter     =    100; // max number of iterations before the clustering forcefully stops

// Generate some random data
let mut samples: Vec<Vec<f64>> = vec![];
for _ in 0..n_samples {
    samples.push((0..n_dimensions).map(|_| rand::random()).collect::<Vec<_>>());
}

// actually perform the clustering
let clustering = kmeans(k, &samples, max_iter);

println!("membership: {:?}", clustering.membership);
println!("centroids : {:?}", clustering.centroids);
```
