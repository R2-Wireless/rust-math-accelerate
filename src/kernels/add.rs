use crate::{AnyResult, BufferType, ConstantBuffer, Dimensions, MathAccelerator};

struct KernelFunctionAdd<'a, T, D>
where
    T: BufferType,
    D: Dimensions,
{
    input_1: &'a ConstantBuffer<'a, T, D>,
    input_2: &'a ConstantBuffer<'a, T, D>,
    pub output: ConstantBuffer<'a, T, D>,
}

impl MathAccelerator {
    pub fn kernel_add<'a, T, D>(
        &'a self,
        input_1: &'a ConstantBuffer<T, D>,
        input_2: &'a ConstantBuffer<T, D>,
    ) -> AnyResult<KernelFunctionAdd<'a, T, D>>
    where
        T: BufferType,
        D: Dimensions,
    {
        assert!(
            input_1.dimensions == input_2.dimensions,
            "Got inputs of different dimensions"
        );
        let me = KernelFunctionAdd {
            input_1,
            input_2,
            output: input_2.duplicate()?,
        };
        let kernel = self.create_kernel(
            "kernel_add",
            me.output.dimensions,
            &[
                ("input_1", me.input_1),
                ("input_2", me.input_2),
                ("output", &me.output),
            ],
        )?;
        kernel.ocl_kernel.enq()?;
        Ok(me)
    }
}
