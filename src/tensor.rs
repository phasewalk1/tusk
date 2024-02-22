use std::ops::{Add, Mul};

trait TensorType: Add + Mul + Default {}
impl TensorType for f32 {}
impl TensorType for f64 {}
impl TensorType for i32 {}
impl TensorType for i64 {}

#[derive(Debug)]
pub struct Tensor<T> where T: TensorType {
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    grad: Option<Box<Tensor<T>>>,
    op: Option<Box<dyn Operation<T>>>,
}

pub trait Operation<T> where T: TensorType, Self: core::fmt::Debug {
    fn forward(&self) -> Tensor<T>;
    fn backward(&self, grad: Tensor<T>) -> Vec<Tensor<T>>;
}

impl<T> Tensor<T> where T: TensorType {
    pub fn new(data: Vec<T>, shape: Vec<usize>, strides: Vec<usize>) -> Self {
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

#[macro_export]
macro_rules! zeros {
    ($type:ty, $shape:expr) => {
        {
            let size: usize = $shape.iter().product();
            let data = vec![<$type>::default(); size];
            let strides = Tensor::<$type>::calculate_strides(&$shape.to_vec());
            Tensor::<$type>::new(data, $shape.to_vec(), strides)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let tf32: Tensor<f32> = zeros!(f32, [2, 3]);
        let tf64: Tensor<f64> = zeros!(f64, [2, 3]);
        let ti32: Tensor<i32> = zeros!(i32, [2, 3]);
    }
}