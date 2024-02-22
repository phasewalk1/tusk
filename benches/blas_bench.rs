use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array2;

fn blas_matmult_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("BlasMatMult");

    // Setting up the matrices for benchmarking
    let matrix_size = 1024; // For 512x512 matrices
    let a = Array2::<f32>::ones((matrix_size, matrix_size));
    let b = Array2::<f32>::ones((matrix_size, matrix_size));
    let mut c = Array2::<f32>::zeros((matrix_size, matrix_size));

    group.bench_function("BLAS f32 512x512 matrices", |bencher| {
        bencher.iter(|| {
            tusk::ops::blas_matmul(&a.view(), &b.view(), &mut c.view_mut());
            black_box(c.view());
        })
    });

    group.finish();
}

// Make sure to include your new benchmark in the criterion group
criterion_group!(benches, blas_matmult_benchmark);
criterion_main!(benches);
