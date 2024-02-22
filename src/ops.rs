use crate::tensor::{Tensor, TensorType};

impl<T> Tensor<T> where T: TensorType + std::ops::Add<Output = T> + Clone {
    pub fn add(&self, other: &Tensor<T>) -> Tensor<T> {
        if self.shape != other.shape {
            panic!("Shapes do not match");
        }

        let data: Vec<T> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();
        Tensor::new(data, self.shape.clone())
    }
}

// will want some way to compute this dynamically from cache line sizes and such
const TILE_SIZE: usize = 32;

impl<T> Tensor<T> where T: TensorType + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy + Default {
    pub fn matmul(&self, other: &Tensor<T>) -> Tensor<T> {
        assert_eq!(self.shape[1], other.shape[0], "Matrices have incompatible shapes");

        let n = self.shape[0];
        let m = other.shape[1];
        let k = self.shape[1];

        let mut result = Tensor::new(vec![T::default(); n * m], vec![n, m]);

        for i in (0..n).step_by(TILE_SIZE) {
            for j in (0..m).step_by(TILE_SIZE) {
                for z in (0..k).step_by(TILE_SIZE) {
                    let tile_end_i = std::cmp::min(i + TILE_SIZE, n);
                    let tile_end_j = std::cmp::min(j + TILE_SIZE, m);
                    let tile_end_z = std::cmp::min(z + TILE_SIZE, k);

                    for x in i..tile_end_i {
                        for y in j..tile_end_j {
                            let mut sum = T::default();
                            for kk in z..tile_end_z {
                                sum = sum + self.data[x * k + kk] * other.data[kk * m + y];
                            }
                            result.data[x * m + y] = result.data[x * m + y] + sum;
                        }
                    }
                }
            }
        }

        result
    }
}