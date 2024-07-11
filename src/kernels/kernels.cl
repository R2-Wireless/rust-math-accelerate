size_t get_global_flat_id() {
    size_t dims = get_work_dim();
    size_t id1 = get_global_id(0);
    size_t size1 = get_global_size(0);

    if (dims == 1) {
        return id1;
    } else if (dims == 2) {
        size_t id2 = get_global_id(1);
        return id1 + id2 * size1;
    }
    size_t size2 = get_global_size(1);
    size_t id2 = get_global_id(1);
    size_t id3 = get_global_id(2);
    return id1 + id2 * size1 + id3 * size2 * size1;
}

__kernel void kernel_add(
    __global const int * const input_1,
    __global const int * const input_2,
    __global int * const output
) {
    size_t id = get_global_flat_id();
    output[id] = input_1[id] + input_2[id];
}
