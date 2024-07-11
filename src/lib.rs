extern crate ocl;

use ocl::{Buffer as OclBuffer, Kernel as OclKernel, SpatialDims as OclDimensions};

type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

const KERNELS_SRC: &str = include_str!("./kernels/kernels.cl");

#[path = "./kernels/index.rs"]
mod kernels;

pub struct MathAccelerator {
    queue: ocl::Queue,
    program: ocl::Program,
}

impl MathAccelerator {
    pub fn new() -> AnyResult<Self> {
        let platform = ocl::Platform::default();
        let devices = ocl::Device::list_all(platform)?;
        println!("Number of devices: {}", devices.len());
        let pro_que = ocl::ProQue::builder()
            .device(devices[devices.len() - 1])
            .src(KERNELS_SRC)
            .dims(1 << 30)
            .build()?;
        let program = pro_que.program().clone();
        let queue = pro_que.queue().clone();

        Ok(Self { program, queue })
    }

    pub fn create_constant_buffer<T>(&self, vec: &[T]) -> AnyResult<ConstantBuffer<T, usize>>
    where
        T: BufferType,
    {
        self.create_constant_buffer_with_dimensions(vec, vec.len())
    }
    pub fn create_constant_buffer_with_dimensions<T, D>(
        &self,
        vec: &[T],
        dimensions: D,
    ) -> AnyResult<ConstantBuffer<T, D>>
    where
        T: BufferType,
        D: Dimensions,
    {
        let len = dimensions.multiply();
        assert!(len == vec.len(), "dimensions not sized same as dimensions");
        let ocl_buffer = OclBuffer::<T>::builder()
            .queue(self.queue.clone())
            .len(match len == 0 {
                true => 1,
                false => len,
            })
            .flags(ocl::flags::MEM_READ_WRITE)
            .build()?;
        Ok(ConstantBuffer {
            math_accelerator: self,
            dimensions,
            ocl_buffer,
        })
    }
    pub fn create_kernel<'a, T, D>(
        &'a self,
        kernel_function_name: &'static str,
        workers: D,
        arguments: &'static [(&'static str, &'a dyn ToKernelArgumentValue<T>)],
    ) -> AnyResult<Kernel>
    where
        T: BufferType,
        D: Dimensions,
    {
        let mut ocl_kernel_builder = OclKernel::builder();
        ocl_kernel_builder
            .program(&self.program)
            .queue(self.queue.clone())
            .global_work_size(workers.global_work_size())
            .name(kernel_function_name);
        for (kernel_argument_name, kernel_argument_value) in arguments.into_iter() {
            let value = kernel_argument_value.to_kernel_argument_value();
            match (value.buffer, value.value) {
                (Some(_), Some(_)) | (None, None) => {
                    panic!("Not possible both values set or not set at all")
                }
                (Some(b), None) => {
                    ocl_kernel_builder.arg_named(*kernel_argument_name, b);
                }
                (None, Some(v)) => {
                    ocl_kernel_builder.arg_named(*kernel_argument_name, v);
                }
            }
        }
        let ocl_kernel = ocl_kernel_builder.build()?;
        Ok(Kernel { ocl_kernel })
    }
}

pub struct Kernel {
    ocl_kernel: OclKernel,
}

impl Kernel {
    pub fn enq(&self) -> AnyResult<()> {
        self.enq()?;
        Ok(())
    }
}
pub struct ConstantBuffer<'a, T, D>
where
    T: BufferType,
    D: Dimensions,
{
    math_accelerator: &'a MathAccelerator,
    dimensions: D,
    ocl_buffer: OclBuffer<T>,
}

impl<'a, T, D> ConstantBuffer<'a, T, D>
where
    T: BufferType,
    D: Dimensions,
{
    pub fn duplicate(&self) -> AnyResult<ConstantBuffer<'a, T, D>> {
        self.duplicate_with_dimensions(self.dimensions)
    }
    pub fn duplicate_with_dimensions<D2>(
        &self,
        dimensions: D2,
    ) -> AnyResult<ConstantBuffer<'a, T, D2>>
    where
        D2: Dimensions,
    {
        let len = dimensions.multiply();
        let ocl_buffer = OclBuffer::<T>::builder()
            .queue(self.math_accelerator.queue.clone())
            .len(match len == 0 {
                true => 1,
                false => len,
            })
            .flags(ocl::flags::MEM_READ_WRITE)
            .build()?;
        Ok(ConstantBuffer {
            math_accelerator: &self.math_accelerator,
            dimensions,
            ocl_buffer,
        })
    }
}

pub trait BufferType:
    Clone + Copy + Default + PartialEq + Send + Sync + 'static + ocl::OclPrm
{
    fn get_vec(size: usize) -> Vec<Self>;
}

impl BufferType for u8 {
    fn get_vec(size: usize) -> Vec<Self> {
        vec![0; size]
    }
}
impl BufferType for i8 {
    fn get_vec(size: usize) -> Vec<Self> {
        vec![0; size]
    }
}
impl BufferType for u16 {
    fn get_vec(size: usize) -> Vec<Self> {
        vec![0; size]
    }
}
impl BufferType for i16 {
    fn get_vec(size: usize) -> Vec<Self> {
        vec![0; size]
    }
}
impl BufferType for u32 {
    fn get_vec(size: usize) -> Vec<Self> {
        vec![0; size]
    }
}
impl BufferType for i32 {
    fn get_vec(size: usize) -> Vec<Self> {
        vec![0; size]
    }
}
impl BufferType for u64 {
    fn get_vec(size: usize) -> Vec<Self> {
        vec![0; size]
    }
}
impl BufferType for i64 {
    fn get_vec(size: usize) -> Vec<Self> {
        vec![0; size]
    }
}
impl BufferType for f32 {
    fn get_vec(size: usize) -> Vec<Self> {
        vec![0f32; size]
    }
}
impl BufferType for f64 {
    fn get_vec(size: usize) -> Vec<Self> {
        vec![0f64; size]
    }
}
impl BufferType for isize {
    fn get_vec(size: usize) -> Vec<Self> {
        vec![0; size]
    }
}
impl BufferType for usize {
    fn get_vec(size: usize) -> Vec<Self> {
        vec![0; size]
    }
}

pub trait Dimensions: Clone + Copy + Default + PartialEq + Send + Sync + 'static {
    fn multiply(&self) -> usize;
    fn global_work_size(&self) -> OclDimensions;
}

impl Dimensions for usize {
    fn multiply(&self) -> usize {
        *self
    }
    fn global_work_size(&self) -> OclDimensions {
        OclDimensions::One(*self)
    }
}
impl Dimensions for (usize, usize) {
    fn multiply(&self) -> usize {
        let (self1, self2) = self;
        self1 * self2
    }
    fn global_work_size(&self) -> OclDimensions {
        let (self1, self2) = self;
        OclDimensions::Two(*self1, *self2)
    }
}
impl Dimensions for (usize, usize, usize) {
    fn multiply(&self) -> usize {
        let (self1, self2, self3) = self;
        self1 * self2 * self3
    }
    fn global_work_size(&self) -> OclDimensions {
        let (self1, self2, self3) = self;
        OclDimensions::Three(*self1, *self2, *self3)
    }
}

trait ToKernelArgumentValue<T>
where
    T: BufferType,
{
    fn to_kernel_argument_value(&self) -> KernelArgumentValue<T>;
}

pub struct KernelArgumentValue<T>
where
    T: BufferType,
{
    buffer: Option<OclBuffer<T>>,
    value: Option<T>,
}

impl<'a, T, D> ToKernelArgumentValue<T> for ConstantBuffer<'a, T, D>
where
    T: BufferType,
    D: Dimensions,
{
    fn to_kernel_argument_value(&self) -> KernelArgumentValue<T> {
        KernelArgumentValue {
            buffer: Some(self.ocl_buffer.clone()),
            value: None,
        }
    }
}

impl<'a, T> ToKernelArgumentValue<T> for T
where
    T: BufferType,
{
    fn to_kernel_argument_value(&self) -> KernelArgumentValue<T> {
        KernelArgumentValue {
            value: Some(*self),
            buffer: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
