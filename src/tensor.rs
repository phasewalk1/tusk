use std::ops::{Add, Mul};
use ndarray::Array2;

pub trait TensorType: Add + Mul + Default + Send + Sync {}
impl TensorType for f32 {}
impl TensorType for f64 {}
impl TensorType for i32 {}
impl TensorType for i64 {}

#[derive(Debug)]
pub struct Tensor<T> where T: TensorType {
    pub data: Array2<T>,
    pub shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    grad: Option<Box<Tensor<T>>>,
    op: Option<Box<dyn Operation<T>>>,
}

pub trait Operation<T: TensorType>: core::fmt::Debug + Send + Sync {
    fn forward(&self) -> Tensor<T>;
    fn backward(&self, grad: Tensor<T>) -> Vec<Tensor<T>>;
}

impl<T> Tensor<T> where T: TensorType {
    pub fn new(data: Array2<T>, shape: Vec<usize>) -> Self {
        let strides = Tensor::<T>::calculate_strides(&shape);
        Tensor {
            data,
            shape,
            strides,
            grad: None,
            op: None,
        }
    }

    pub fn with_shape(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        let data = Array2::<T>::default((size, 1));
        Tensor::<T>::new(data, shape.to_vec())
    }

    fn calculate_strides(shape: &Vec<usize>) -> Vec<usize> {
        let mut strides = Vec::with_capacity(shape.len());
        let mut stride = 1;
        for &dim in shape.iter().rev() {
            strides.push(stride);
            stride *= dim;
        }
        strides.reverse();
        strides
    }
}