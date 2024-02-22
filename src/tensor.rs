use std::ops::{Add, Mul};

pub(crate) trait TensorType: Add + Mul + Default {}
impl TensorType for f32 {}
impl TensorType for f64 {}
impl TensorType for i32 {}
impl TensorType for i64 {}

#[derive(Debug)]
pub struct Tensor<T> where T: TensorType {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    grad: Option<Box<Tensor<T>>>,
    op: Option<Box<dyn Operation<T>>>,
}

pub trait Operation<T> where T: TensorType, Self: core::fmt::Debug {
    fn forward(&self) -> Tensor<T>;
    fn backward(&self, grad: Tensor<T>) -> Vec<Tensor<T>>;
}

impl<T> Tensor<T> where T: TensorType {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        let strides = Tensor::<T>::calculate_strides(&shape);
        Tensor {
            data,
            shape,
            strides,
            grad: None,
            op: None,
        }
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