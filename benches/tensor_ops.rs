use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tusk::{ones, Tensor};

fn tensor_addition_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("TensorAddition");

    let tensor_size = 1024 * 1024; // Example size: 1M elements
    let a = ones!(f32, vec![tensor_size]);
    let b = ones!(f32, vec![tensor_size]);

    group.bench_function("f32 1M elements", |bencher| {
        bencher.iter(|| {
            let result = a.add(&b);
            black_box(result);
        })
    });

    group.finish();
}

fn matmult(c: &mut Criterion) {
    let mut group = c.benchmark_group("MatMult");

    let matrix_size = 512;
    let a = ones!(f32, vec![matrix_size, matrix_size]);
    let b = ones!(f32, vec![matrix_size, matrix_size]);

    group.bench_function("f32 512x512 matrices", |bencher| {
        bencher.iter(|| {
            let result = a.matmul(&b);
            black_box(result);
        })
    });

    group.finish();
}

criterion_group!(benches, tensor_addition_benchmark, matmult);
criterion_main!(benches);