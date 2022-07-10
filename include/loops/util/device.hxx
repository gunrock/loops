namespace loops {
namespace device {
typedef int device_id_t;

void set(device_id_t ordinal) {
  cudaSetDevice(ordinal);
}

device_id_t get() {
  device_id_t ordinal;
  cudaGetDevice(&ordinal);
  return ordinal;
}

struct properties_t {
  typedef cudaDeviceProp device_properties_t;
  device_properties_t properties;
  device_id_t ordinal;

  properties_t() : ordinal(device::get()) {
    cudaGetDeviceProperties(&properties, ordinal);
  }

  int multi_processor_count() { return properties.multiProcessorCount; }
};

}  // namespace device
}  // namespace loops