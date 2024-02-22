#[macro_export]
macro_rules! zeros {
    ($type:ty, $shape:expr) => {
        {
            let size: usize = $shape.iter().product();
            let data = vec![<$type>::default(); size];
            Tensor::<$type>::new(data, $shape.to_vec())
        }
    };
}

#[macro_export]
macro_rules! ones {
    ($type:ty, $shape:expr) => {
        {
            let size: usize = $shape.iter().product();
            let data = vec![<$type>::default() + (1 as $type); size];
            Tensor::<$type>::new(data, $shape.to_vec())
        }
    };
}