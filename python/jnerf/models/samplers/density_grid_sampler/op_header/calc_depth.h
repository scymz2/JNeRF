#include"ray_sampler_header.h"

__device__ float unwarp_dt(float dt, int NERF_CASCADES, float MIN_CONE_STEPSIZE)
{
	float max_stepsize = MIN_CONE_STEPSIZE * (1 << (NERF_CASCADES - 1));
	return dt * (max_stepsize - MIN_CONE_STEPSIZE) + MIN_CONE_STEPSIZE;
}

template <typename TYPE>
__global__ void compute_depths(
	const uint32_t n_rays,						// batch total rays number
	BoundingBox aabb,							// bounding box range
	int padded_output_width,    				// network output width
	const TYPE *network_output, 				// network output
	ENerfActivation density_activation,			// activation of density in output 
	PitchedPtr<NerfCoordinate> coords_in,		// network input (xyz, dt, dir)
	uint32_t *__restrict__ numsteps_in,			// rays offset and base counter before compact
	float *depth_output, 						// rays depth output
	uint32_t *__restrict__ numsteps_compacted_in,// rays offset and base counter after compact
	int NERF_CASCADES,							// num of density grid level
	float MIN_CONE_STEPSIZE						// lower bound of step size
	)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays)
	{
		return;
	}

	uint32_t numsteps = numsteps_compacted_in[i * 2 + 0];
	uint32_t base = numsteps_compacted_in[i * 2 + 1];
	if (numsteps == 0)
	{
		depth_output[i] = 0.0f;
		return;
	}
	coords_in += base;
	network_output += base * padded_output_width;

	float T = 1.f;

	float depth_ray = 0.0f;

	uint32_t compacted_numsteps = 0;
	for (; compacted_numsteps < numsteps; ++compacted_numsteps)
	{
		const vector_t<TYPE, 4> local_network_output = *(vector_t<TYPE, 4> *)network_output;
		const float dt = unwarp_dt(coords_in.ptr->dt, NERF_CASCADES, MIN_CONE_STEPSIZE);

		float density = network_to_density(float(local_network_output[3]), density_activation);

		const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		depth_ray += weight * coords_in.ptr->pos.p.z(); // Assuming z-axis depth, adjust if necessary

		T *= (1.f - alpha);
		network_output += padded_output_width;
		coords_in += 1;
	}

	depth_output[i] = depth_ray;
}

template <typename TYPE>
__global__ void compute_depths_inference(
	const uint32_t n_rays,						// batch total rays number
	BoundingBox aabb,							// bounding box range
	int padded_output_width,					// network output width
	const TYPE *network_output,					// network output
	ENerfActivation density_activation,			// activation of density in output 
	PitchedPtr<NerfCoordinate> coords_in,		// network input (xyz, dt, dir)
	uint32_t *__restrict__ numsteps_in,			// rays offset and base counter
	float *__restrict__ depth_output			// rays depth output
	int NERF_CASCADES,							// num of density grid level
	float MIN_CONE_STEPSIZE						// lower bound of step size
	)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= n_rays)
	{
		return;
	}

	uint32_t numsteps = numsteps_in[i * 2 + 0];
	uint32_t base = numsteps_in[i * 2 + 1];
	if (numsteps == 0)
	{
		depth_output[i] = 0.0f;
		return;
	}
	coords_in += base;
	network_output += base * padded_output_width;

	float T = 1.f;

	float depth_ray = 0.0f;

	uint32_t compacted_numsteps = 0;
	for (; compacted_numsteps < numsteps; ++compacted_numsteps)
	{
		const vector_t<TYPE, 4> local_network_output = *(vector_t<TYPE, 4> *)network_output;
		const float dt = unwarp_dt(coords_in.ptr->dt, NERF_CASCADES, MIN_CONE_STEPSIZE);

		float density = network_to_density(float(local_network_output[3]), density_activation);

		const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		depth_ray += weight * coords_in.ptr->pos.p.z(); // Assuming z-axis depth, adjust if necessary

		T *= (1.f - alpha);
		network_output += padded_output_width;
		coords_in += 1;
	}
	depth_output[i] = depth_ray;
}
