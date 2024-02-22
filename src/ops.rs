use ndarray::{ArrayView2, ArrayViewMut2};
use ndarray::linalg::general_mat_mul;

pub fn blas_matmul<T>(a: &ArrayView2<T>, b: &ArrayView2<T>, c: &mut ArrayViewMut2<T>)
where
    T: ndarray::LinalgScalar,
{
    general_mat_mul(T::one(), a, b, T::zero(), c);
}