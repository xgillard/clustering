
// Copyright 2022 Xavier Gillard
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! This crate provides an easy and efficient way to perform kmeans 
//! clustering on arbitrary data. The algo is initialized with kmeans++
//! for best performance of the clustering. 
//! 
//! There are three goals to this implementation of the kmeans algorithm:
//! 
//! 1. it must be generic
//! 2. it must be easy to use
//! 3. it must be reasonably fast
//! 
//! # Example
//! ```
//! 
//! use clustering::*;
//! 
//! let n_samples    = 20_000; // # of samples in the example
//! let n_dimensions =    200; // # of dimensions in each sample
//! let k            =      4; // # of clusters in the result
//! let max_iter     =    100; // max number of iterations before the clustering forcefully stops
//! 
//! // Generate some random data
//! let mut samples: Vec<Vec<f64>> = vec![];
//! for _ in 0..n_samples {
//!     samples.push((0..n_dimensions).map(|_| rand::random()).collect::<Vec<_>>());
//! }
//! 
//! // actually perform the clustering
//! let clustering = kmeans(k, &samples, max_iter);
//! 
//! println!("membership: {:?}", clustering.membership);
//! println!("centroids : {:?}", clustering.centroids);
//! ```

#[cfg(feature = "parallel")]
use std::cmp::Ordering;
#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicUsize,AtomicU64};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "logging")]
use lazy_static::lazy_static;

//==========================================================================

// install a logger facility
#[cfg(feature = "logging")]
lazy_static! {
    static ref _LOG: () = env_logger::init();
}

//=============================================================================

/// This is the trait you will want to implement for the types you wish to cluster.
pub trait Elem {
    /// This is the number of dimensions (aka features) of the elements you wish to
    /// cluster using kmeans
    fn dimensions(&self) -> usize;
    /// This returns the ith dimention associated with the given element.
    fn at(&self, i: usize) -> f64;
}

/// A centroid: a collection of n abstract quantities (which must be interpreted
/// in the context of what *you* are doing).
#[derive(Debug)]
pub struct Centroid(pub Vec<f64>);
/// This is the result of a kmeans clustering
pub struct Clustering<'a, T> {
    /// The set of elements that have been clustered (in that order)
    pub elements: &'a [T],
    /// The membership assignment
    /// membership[i] = y means that element[i] belongs to cluster y
    pub membership: Vec<usize>,
    /// The centroids of the clusters in this given clustering
    pub centroids: Vec<Centroid>,
}

#[cfg(feature = "parallel")]
/// This function returns a clustering that groups the given set of 
/// 'elems' in 'k' clusters and will at most perform 'iter' iterations before stopping
pub fn kmeans<T: Elem + Sync>(k: usize, elems: &[T], iter: usize) -> Clustering<T> {
    let mut centroids = initialize(k, elems);
    let membership : Vec<AtomicUsize> = (0..elems.len())
            .map(|_| AtomicUsize::new(0usize))
            .collect();
    let mut counts = vec![0; k];

    #[allow(unused_variables)] // -> it can be used if logging is enabled
    for it in 0..iter {
        let changes = AtomicU64::new(0);

        // assign each vertex to the cluster whose centroid is the closest
        let dispatch_element = |i : usize| -> usize {
            let e = &elems[i];
            let old = membership[i].load(std::sync::atomic::Ordering::SeqCst);
            let dist = square_distance(e, &centroids[old]);

            let (best_c, best_d) : (usize, f64) = (0..centroids.len())
                    .map(|c| (c, square_distance(e, &centroids[c])))
                    .min_by(|(_c1,d1), (_c2, d2)| if d1 < d2 
                                {Ordering::Less} 
                            else 
                                {Ordering::Greater })
                    .unwrap();
            // we have nearest center
            if best_c != old {
                changes.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                membership[i].store(best_c, std::sync::atomic::Ordering::SeqCst);
            }
            assert!(best_d <= dist); 
            best_c           
        };

        let _res : Vec<usize> = (0..elems.len()).into_par_iter().map(dispatch_element).collect();

        // recompute the n-dimensions of each centroid
        // -> start resetting all centroid data
        counts.iter_mut().for_each(|x| *x = 0);
        centroids.iter_mut().for_each(|c| 
            c.0.iter_mut().for_each(|d| *d = 0.0));
        
        for (i, elem) in elems.iter().enumerate() {
            let clus = membership[i].load(std::sync::atomic::Ordering::SeqCst);
            counts[clus] += 1;
            
            for (d, dim) in centroids[clus].0.iter_mut().enumerate() {
                *dim += elem.at(d);
            }
        }
        
        // -> normalize the computed distances
        for (centroid, size) in centroids.iter_mut().zip(counts.iter().copied()) {
            centroid.0.iter_mut().for_each(|d| if size == 0 { *d = 0.0 } else {*d /= size as f64});
        }
                
        // short circuit
        if changes.load(std::sync::atomic::Ordering::SeqCst) == 0 {
            #[cfg(feature = "logging")]
            log::info!("clustering kmeans: short circuit after nb iter : {}", it);
            break;
        }
    }

    Clustering { 
        elements: elems, 
        membership : membership.iter().map(|x| x.load(std::sync::atomic::Ordering::SeqCst)).collect::<Vec<usize>>(), 
        centroids
    }
}

#[cfg(not(feature = "parallel"))]
/// This function returns a clustering that groups the given set of 
/// 'elems' in 'k' clusters and will at most perform 'iter' iterations before stopping
pub fn kmeans<T: Elem>(k: usize, elems: &[T], iter: usize) -> Clustering<T> {
    let mut centroids = initialize(k, elems);
    let mut membership = vec![0; elems.len()];
    let mut counts = vec![0; k];

    #[allow(unused_variables)] // -> it can be used if logging is enabled
    for it in 0..iter {
        let mut changes = 0;

        // assign each vertex to the cluster whose centroid is the closest
        for (i, e) in elems.iter().enumerate() {
            let old = membership[i];
            let mut clus = old;
            let mut dist = square_distance(e, &centroids[old]);

            for (c, centroid) in centroids.iter().enumerate() {
                let sdist = square_distance(e, centroid);
                if sdist < dist {
                    dist = sdist;
                    clus = c;
                    changes += 1;
                }
            }

            membership[i] = clus;
        }

        // recompute the n-dimensions of each centroid
        // -> start resetting all centroid data
        counts.iter_mut().for_each(|x| *x = 0);
        centroids.iter_mut().for_each(|c| 
            c.0.iter_mut().for_each(|d| *d = 0.0));
        
        for (i, elem) in elems.iter().enumerate() {
            let clus = membership[i];
            counts[clus] += 1;
            
            for (d, dim) in centroids[clus].0.iter_mut().enumerate() {
                *dim += elem.at(d);
            }
        }
        
        // -> normalize the computed distances
        for (centroid, size) in centroids.iter_mut().zip(counts.iter().copied()) {
            centroid.0.iter_mut().for_each(|d| if size == 0 { *d = 0.0 } else {*d /= size as f64});
        }
                
        // short circuit
        if changes == 0 {
            #[cfg(feature = "logging")]
            log::info!("clustering kmeans: short circuit after nb iter : {}", it);
            break;
        }
    }

    Clustering { 
        elements: elems, 
        membership, 
        centroids
    }
}

//- /// Returns the generalized euclidean distance between elements a and b
//- fn distance(a: &dyn Elem, b: &dyn Elem) -> f64 {
//-    square_distance(a, b).sqrt()
//- }

/// Returns the squared generalized euclidean distance between elements 
/// a and b. (for performance reasons, you will want to use that info instead
/// of the actual 'distance' function).
fn square_distance(a: &dyn Elem, b: &dyn Elem) -> f64 {
    let mut tot = 0.0;
    let n = a.dimensions();
    for i in 0..n {
        let dim = b.at(i) - a.at(i);
        tot += dim * dim;
    }
    tot
}

/// This method performs a kmeans++ initialization. 
/// It returns a vector of centroids that are all equal to one of the vertices
/// and all the centroids have greedily been chosen to be as far from one another
/// as possibly can
fn initialize<T: Elem>(k: usize, elems: &[T]) -> Vec<Centroid> {
    let mut taken = vec![false; elems.len()];
    let mut centroids = vec![];

    let first = rand::random::<usize>() % elems.len();
    taken[first] = true;
    centroids.push(new_centroid(&elems[first]));

    for _ in 1..k {
        let mut imax = 0;
        let mut dmax = f64::NEG_INFINITY;

        // take the remaining elem that is the farthest apart from its closest centroid
        for (i, elem) in elems.iter().enumerate() {
            if taken[i] {
                continue;
            }
            
            let mut dxmin = f64::INFINITY;
            for centroid in centroids.iter() {
                let dx = square_distance(elem, centroid);

                if dx < dxmin {
                    dxmin = dx;
                }
            }

            if dxmin > dmax {
                dmax = dxmin;
                imax = i;
            }
        }
        
        taken[imax] = true;
        centroids.push(new_centroid(&elems[imax]));
    }

    centroids
}

/// Utility function to create a centroid from the given element
fn new_centroid<T: Elem>(elem: &T) -> Centroid {
    let mut centroid = vec![];
    let dimensions = elem.dimensions();
    for i in 0..dimensions {
        centroid.push(elem.at(i));
    }
    Centroid(centroid)
}

/// A centroid is considered to be an element
impl Elem for Centroid {
    fn dimensions(&self) -> usize {
        self.0.len()
    }

    fn at(&self, i: usize)  -> f64 {
        self.0[i]
    }
}

/// Implementation of the element trait for primitive vectors and slices
macro_rules! elem {
    ($x: ty) => {
        impl Elem for Vec<$x> {
            fn dimensions(&self) -> usize {
                self.len()
            }
        
            fn at(&self, i: usize)  -> f64 {
                self[i] as f64
            }
        }
        impl Elem for &[$x] {
            fn dimensions(&self) -> usize {
                self.len()
            }
        
            fn at(&self, i: usize)  -> f64 {
                self[i] as f64
            }
        }
    };
}

elem!(u8);
elem!(u16);
elem!(u32);
elem!(u64);
elem!(usize);

elem!(i8);
elem!(i16);
elem!(i32);
elem!(i64);
elem!(isize);

elem!(f32);
elem!(f64);

#[cfg(test)]
mod test {
    use crate::*;

    #[test]
    fn test_impl() {
        let a: &[i32] = &[1, 2, 3, 4];
        let b: &[i32] = &[5, 4, 3, 2];

        let dst = square_distance(&a, &b);
        assert_eq!(24.0, dst);
    }

    #[test]
    fn example() {
        let items: &[&[f32]] = &[
            //
            &[ 1.0],
            &[ 1.1],
            &[ 0.9],
            //
            &[10.0],
            &[11.1],
            &[10.9],

            //
            &[30.0],
            &[31.1],
            &[30.9],
        ];

        let clus = kmeans(3, items, 1000);
        println!("centroids  = {:?}", clus.membership);
        println!("membership = {:?}", clus.centroids);
    }
}