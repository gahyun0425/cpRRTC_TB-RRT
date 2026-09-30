// Internal device-environment ownership and per-request reset implementation.
// Included by PATACON.cu inside namespace PATACON.

    inline void setup_environment_on_device(ppln::collision::Environment<float> *&d_env, const ppln::collision::Environment<float> &h_env) {
        // allocate the environment struct
        cudaMalloc(&d_env, sizeof(ppln::collision::Environment<float>));
        // Initialize struct to zeros first
        cudaMemset(d_env, 0, sizeof(ppln::collision::Environment<float>));

        // Handle each primitive type separately
        if (h_env.num_spheres > 0) {
            // Allocate and copy spheres array
            ppln::collision::Sphere<float> *d_spheres;
            cudaMalloc(&d_spheres, sizeof(ppln::collision::Sphere<float>) * h_env.num_spheres);
            cudaMemcpy(d_spheres, h_env.spheres, sizeof(ppln::collision::Sphere<float>) * h_env.num_spheres, cudaMemcpyHostToDevice);
            // Update the struct fields directly
            cudaMemcpy(&(d_env->spheres), &d_spheres, sizeof(ppln::collision::Sphere<float>*), cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_spheres), &h_env.num_spheres, sizeof(unsigned int), cudaMemcpyHostToDevice);
        }

        if (h_env.num_capsules > 0) {
            ppln::collision::Capsule<float> *d_capsules;
            cudaMalloc(&d_capsules, sizeof(ppln::collision::Capsule<float>) * h_env.num_capsules);
            cudaMemcpy(d_capsules, h_env.capsules,sizeof(ppln::collision::Capsule<float>) * h_env.num_capsules,cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->capsules), &d_capsules, sizeof(ppln::collision::Capsule<float>*),cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_capsules), &h_env.num_capsules, sizeof(unsigned int),cudaMemcpyHostToDevice);
        }

        // Repeat for each primitive type...
        if (h_env.num_z_aligned_capsules > 0) {
            ppln::collision::Capsule<float> *d_z_capsules;
            cudaMalloc(&d_z_capsules, sizeof(ppln::collision::Capsule<float>) * h_env.num_z_aligned_capsules);
            cudaMemcpy(d_z_capsules, h_env.z_aligned_capsules,sizeof(ppln::collision::Capsule<float>) * h_env.num_z_aligned_capsules,cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->z_aligned_capsules), &d_z_capsules, sizeof(ppln::collision::Capsule<float>*),cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_z_aligned_capsules), &h_env.num_z_aligned_capsules, sizeof(unsigned int),cudaMemcpyHostToDevice);
        }

        if (h_env.num_cylinders > 0) {
            ppln::collision::Cylinder<float> *d_cylinders;
            cudaMalloc(&d_cylinders, sizeof(ppln::collision::Cylinder<float>) * h_env.num_cylinders);
            cudaMemcpy(d_cylinders, h_env.cylinders,sizeof(ppln::collision::Cylinder<float>) * h_env.num_cylinders,cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->cylinders), &d_cylinders, sizeof(ppln::collision::Cylinder<float>*),cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_cylinders), &h_env.num_cylinders, sizeof(unsigned int),cudaMemcpyHostToDevice);
        }

        if (h_env.num_cuboids > 0) {
            ppln::collision::Cuboid<float> *d_cuboids;
            cudaMalloc(&d_cuboids, sizeof(ppln::collision::Cuboid<float>) * h_env.num_cuboids);
            cudaMemcpy(d_cuboids, h_env.cuboids,sizeof(ppln::collision::Cuboid<float>) * h_env.num_cuboids,cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->cuboids), &d_cuboids, sizeof(ppln::collision::Cuboid<float>*),cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_cuboids), &h_env.num_cuboids, sizeof(unsigned int),cudaMemcpyHostToDevice);
        }

        if (h_env.num_z_aligned_cuboids > 0) {
            ppln::collision::Cuboid<float> *d_z_cuboids;
            cudaMalloc(&d_z_cuboids, sizeof(ppln::collision::Cuboid<float>) * h_env.num_z_aligned_cuboids);
            cudaMemcpy(d_z_cuboids, h_env.z_aligned_cuboids,sizeof(ppln::collision::Cuboid<float>) * h_env.num_z_aligned_cuboids,cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(d_env->z_aligned_cuboids), &d_z_cuboids, sizeof(ppln::collision::Cuboid<float>*),cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_z_aligned_cuboids), &h_env.num_z_aligned_cuboids, sizeof(unsigned int),cudaMemcpyHostToDevice);
        }
    }


    inline void cleanup_environment_on_device(ppln::collision::Environment<float> *d_env, const ppln::collision::Environment<float> &h_env) {
        // Get the pointers from device struct before freeing
        ppln::collision::Sphere<float> *d_spheres = nullptr;
        ppln::collision::Capsule<float> *d_capsules = nullptr;
        ppln::collision::Capsule<float> *d_z_capsules = nullptr;
        ppln::collision::Cylinder<float> *d_cylinders = nullptr;
        ppln::collision::Cuboid<float> *d_cuboids = nullptr;
        ppln::collision::Cuboid<float> *d_z_cuboids = nullptr;

        // Copy each pointer from device memory
        if (h_env.num_spheres > 0) {
            cudaMemcpy(&d_spheres, &(d_env->spheres), sizeof(ppln::collision::Sphere<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_spheres);
        }
        
        if (h_env.num_capsules > 0) {
            cudaMemcpy(&d_capsules, &(d_env->capsules), sizeof(ppln::collision::Capsule<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_capsules);
        }
        
        if (h_env.num_z_aligned_capsules > 0) {
            cudaMemcpy(&d_z_capsules, &(d_env->z_aligned_capsules), sizeof(ppln::collision::Capsule<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_z_capsules);
        }
        
        if (h_env.num_cylinders > 0) {
            cudaMemcpy(&d_cylinders, &(d_env->cylinders), sizeof(ppln::collision::Cylinder<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_cylinders);
        }
        
        if (h_env.num_cuboids > 0) {
            cudaMemcpy(&d_cuboids, &(d_env->cuboids), sizeof(ppln::collision::Cuboid<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_cuboids);
        }
        
        if (h_env.num_z_aligned_cuboids > 0) {
            cudaMemcpy(&d_z_cuboids, &(d_env->z_aligned_cuboids), sizeof(ppln::collision::Cuboid<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_z_cuboids);
        }

        // Finally free the environment struct itself
        cudaFree(d_env);
    }

    // Keep fixed world primitives resident across replans. update() transfers
    // only arrays whose host contents changed (normally the moving sphere).
    class PersistentDeviceEnvironment {
    public:
        ~PersistentDeviceEnvironment() {
            release();
        }

        ppln::collision::Environment<float> *update(
            const ppln::collision::Environment<float> &host
        ) {
            if (!matches(host)) {
                release();
                allocate(host);
            } else {
                update_buffer(device_spheres_, host.spheres, host.num_spheres,
                              cached_spheres_);
                update_buffer(device_capsules_, host.capsules,
                              host.num_capsules, cached_capsules_);
                update_buffer(device_z_aligned_capsules_,
                              host.z_aligned_capsules,
                              host.num_z_aligned_capsules,
                              cached_z_aligned_capsules_);
                update_buffer(device_cylinders_, host.cylinders,
                              host.num_cylinders, cached_cylinders_);
                update_buffer(device_cuboids_, host.cuboids,
                              host.num_cuboids, cached_cuboids_);
                update_buffer(device_z_aligned_cuboids_,
                              host.z_aligned_cuboids,
                              host.num_z_aligned_cuboids,
                              cached_z_aligned_cuboids_);
            }
            return device_environment_;
        }

    private:
        static bool same_device_primitive(
            const ppln::collision::Sphere<float> &left,
            const ppln::collision::Sphere<float> &right
        ) {
            return left.min_distance == right.min_distance
                && left.x == right.x && left.y == right.y
                && left.z == right.z && left.r == right.r;
        }

        static bool same_device_primitive(
            const ppln::collision::Cylinder<float> &left,
            const ppln::collision::Cylinder<float> &right
        ) {
            return left.min_distance == right.min_distance
                && left.x1 == right.x1 && left.y1 == right.y1
                && left.z1 == right.z1 && left.xv == right.xv
                && left.yv == right.yv && left.zv == right.zv
                && left.r == right.r && left.rdv == right.rdv;
        }

        static bool same_device_primitive(
            const ppln::collision::Cuboid<float> &left,
            const ppln::collision::Cuboid<float> &right
        ) {
            return left.min_distance == right.min_distance
                && left.x == right.x && left.y == right.y
                && left.z == right.z
                && left.axis_1_x == right.axis_1_x
                && left.axis_1_y == right.axis_1_y
                && left.axis_1_z == right.axis_1_z
                && left.axis_2_x == right.axis_2_x
                && left.axis_2_y == right.axis_2_y
                && left.axis_2_z == right.axis_2_z
                && left.axis_3_x == right.axis_3_x
                && left.axis_3_y == right.axis_3_y
                && left.axis_3_z == right.axis_3_z
                && left.axis_1_r == right.axis_1_r
                && left.axis_2_r == right.axis_2_r
                && left.axis_3_r == right.axis_3_r;
        }

        template <typename Primitive>
        static void update_buffer(
            Primitive *device,
            const Primitive *host,
            unsigned int count,
            std::vector<Primitive> &cached
        ) {
            const std::size_t bytes =
                static_cast<std::size_t>(count) * sizeof(Primitive);
            if (cached.size() != count) {
                if (bytes > 0) {
                    cudaCheckError(cudaMemcpy(
                        device, host, bytes, cudaMemcpyHostToDevice
                    ));
                }
                cached.assign(host, host + count);
                return;
            }

            // Primitive names live only on the host. Compare the numeric
            // collision payload and upload only elements that changed, so a
            // moving sphere does not re-upload static world geometry.
            for (unsigned int index = 0; index < count; ++index) {
                if (same_device_primitive(cached[index], host[index])) {
                    continue;
                }
                cudaCheckError(cudaMemcpy(
                    device + index,
                    host + index,
                    sizeof(Primitive),
                    cudaMemcpyHostToDevice
                ));
                cached[index] = host[index];
            }
        }

        template <typename Primitive>
        static void allocate_buffer(
            Primitive *&device,
            const Primitive *host,
            unsigned int count,
            std::vector<Primitive> &cached
        ) {
            if (count == 0) {
                device = nullptr;
                cached.clear();
                return;
            }
            cudaCheckError(cudaMalloc(
                &device,
                static_cast<std::size_t>(count) * sizeof(Primitive)
            ));
            cached.clear();
            update_buffer(device, host, count, cached);
        }

        template <typename Primitive>
        static void free_buffer(Primitive *&device) noexcept {
            if (device != nullptr) {
                cudaFree(device);
                device = nullptr;
            }
        }

        bool matches(
            const ppln::collision::Environment<float> &host
        ) const {
            return device_environment_ != nullptr
                && num_spheres_ == host.num_spheres
                && num_capsules_ == host.num_capsules
                && num_z_aligned_capsules_ == host.num_z_aligned_capsules
                && num_cylinders_ == host.num_cylinders
                && num_cuboids_ == host.num_cuboids
                && num_z_aligned_cuboids_ == host.num_z_aligned_cuboids;
        }

        void allocate(const ppln::collision::Environment<float> &host) {
            num_spheres_ = host.num_spheres;
            num_capsules_ = host.num_capsules;
            num_z_aligned_capsules_ = host.num_z_aligned_capsules;
            num_cylinders_ = host.num_cylinders;
            num_cuboids_ = host.num_cuboids;
            num_z_aligned_cuboids_ = host.num_z_aligned_cuboids;

            cudaCheckError(cudaMalloc(
                &device_environment_,
                sizeof(ppln::collision::Environment<float>)
            ));
            cudaCheckError(cudaMemset(
                device_environment_, 0,
                sizeof(ppln::collision::Environment<float>)
            ));
            allocate_buffer(device_spheres_, host.spheres, host.num_spheres,
                            cached_spheres_);
            allocate_buffer(device_capsules_, host.capsules,
                            host.num_capsules, cached_capsules_);
            allocate_buffer(device_z_aligned_capsules_,
                            host.z_aligned_capsules,
                            host.num_z_aligned_capsules,
                            cached_z_aligned_capsules_);
            allocate_buffer(device_cylinders_, host.cylinders,
                            host.num_cylinders, cached_cylinders_);
            allocate_buffer(device_cuboids_, host.cuboids,
                            host.num_cuboids, cached_cuboids_);
            allocate_buffer(device_z_aligned_cuboids_,
                            host.z_aligned_cuboids,
                            host.num_z_aligned_cuboids,
                            cached_z_aligned_cuboids_);

#define INSTALL_ENV_FIELD(field, device_value, count_field, count_value)    \
            cudaCheckError(cudaMemcpy(                                      \
                &(device_environment_->field), &device_value,               \
                sizeof(device_value), cudaMemcpyHostToDevice                \
            ));                                                              \
            cudaCheckError(cudaMemcpy(                                      \
                &(device_environment_->count_field), &count_value,           \
                sizeof(count_value), cudaMemcpyHostToDevice                  \
            ))
            INSTALL_ENV_FIELD(spheres, device_spheres_, num_spheres,
                              num_spheres_);
            INSTALL_ENV_FIELD(capsules, device_capsules_, num_capsules,
                              num_capsules_);
            INSTALL_ENV_FIELD(z_aligned_capsules,
                              device_z_aligned_capsules_,
                              num_z_aligned_capsules,
                              num_z_aligned_capsules_);
            INSTALL_ENV_FIELD(cylinders, device_cylinders_, num_cylinders,
                              num_cylinders_);
            INSTALL_ENV_FIELD(cuboids, device_cuboids_, num_cuboids,
                              num_cuboids_);
            INSTALL_ENV_FIELD(z_aligned_cuboids,
                              device_z_aligned_cuboids_,
                              num_z_aligned_cuboids,
                              num_z_aligned_cuboids_);
#undef INSTALL_ENV_FIELD
        }

        void release() noexcept {
            free_buffer(device_spheres_);
            free_buffer(device_capsules_);
            free_buffer(device_z_aligned_capsules_);
            free_buffer(device_cylinders_);
            free_buffer(device_cuboids_);
            free_buffer(device_z_aligned_cuboids_);
            if (device_environment_ != nullptr) {
                cudaFree(device_environment_);
                device_environment_ = nullptr;
            }
            num_spheres_ = 0;
            num_capsules_ = 0;
            num_z_aligned_capsules_ = 0;
            num_cylinders_ = 0;
            num_cuboids_ = 0;
            num_z_aligned_cuboids_ = 0;
        }

        ppln::collision::Environment<float> *device_environment_ = nullptr;
        ppln::collision::Sphere<float> *device_spheres_ = nullptr;
        ppln::collision::Capsule<float> *device_capsules_ = nullptr;
        ppln::collision::Capsule<float> *device_z_aligned_capsules_ = nullptr;
        ppln::collision::Cylinder<float> *device_cylinders_ = nullptr;
        ppln::collision::Cuboid<float> *device_cuboids_ = nullptr;
        ppln::collision::Cuboid<float> *device_z_aligned_cuboids_ = nullptr;
        unsigned int num_spheres_ = 0;
        unsigned int num_capsules_ = 0;
        unsigned int num_z_aligned_capsules_ = 0;
        unsigned int num_cylinders_ = 0;
        unsigned int num_cuboids_ = 0;
        unsigned int num_z_aligned_cuboids_ = 0;
        std::vector<ppln::collision::Sphere<float>> cached_spheres_;
        std::vector<ppln::collision::Capsule<float>> cached_capsules_;
        std::vector<ppln::collision::Capsule<float>> cached_z_aligned_capsules_;
        std::vector<ppln::collision::Cylinder<float>> cached_cylinders_;
        std::vector<ppln::collision::Cuboid<float>> cached_cuboids_;
        std::vector<ppln::collision::Cuboid<float>> cached_z_aligned_cuboids_;
    };

    static __global__ void reset_device_variables_kernel() {
        solved = 0;
        atomic_free_index[0] = 0;
        atomic_free_index[1] = 0;
        nodes_size[0] = 0;
        nodes_size[1] = 0;
        completed_nodes[0] = 0;
        completed_nodes[1] = 0;
        path_size[0] = 0;
        path_size[1] = 0;
        cost = 0.0f;
        reached_goal_idx = 0;
        connection_tree_id = -1;
        connection_node_idx = -1;
        connection_other_tree_id = -1;
        connection_other_node_idx = -1;
    }

    inline void reset_device_variables() {
        reset_device_variables_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
        }
    }

    __device__ __forceinline__ void reset_to_unwritten_state(volatile float *buffer, int size, int tid) {
        if (tid == 0) {
            for (int i = 0; i < size; i++) {
                buffer[i] = UNWRITTEN_VAL;
            }
        }
        __syncthreads();
    }
