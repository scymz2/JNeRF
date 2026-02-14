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
	float *__restrict__ depth_output,			// rays depth output
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

template <typename TYPE>
__global__ void compute_depths_grad(
	const uint32_t n_rays,
	BoundingBox aabb,
	int padded_output_width,
	TYPE *__restrict__ dloss_doutput,
	const TYPE *network_output,
	uint32_t *__restrict__ numsteps_compacted_in,
	PitchedPtr<NerfCoordinate> coords_in,
	ENerfActivation density_activation,
	const float *__restrict__ grad_x,
	float *__restrict__ density_grid_mean,
	int NERF_CASCADES,
	float MIN_CONE_STEPSIZE
	)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays)
	{
		return;
	}
	float loss_scale = 128;
	loss_scale /= n_rays;
	uint32_t numsteps = numsteps_compacted_in[i * 2 + 0];
	uint32_t base = numsteps_compacted_in[i * 2 + 1];

	coords_in += base;
	network_output += base * padded_output_width;
	dloss_doutput += base * padded_output_width;

	const float dloss_by_ddepth = grad_x[i];

	const float output_l1_reg_density = *density_grid_mean < NERF_MIN_OPTICAL_THICKNESS() ? 1e-4f : 0.0f;

	// First pass: recompute forward to get total depth_ray
	float T = 1.f;
	float depth_ray = 0.0f;

	const TYPE *net_out_fwd = network_output;
	PitchedPtr<NerfCoordinate> coords_fwd = coords_in;
	for (uint32_t s = 0; s < numsteps; ++s)
	{
		const vector_t<TYPE, 4> local_network_output = *(vector_t<TYPE, 4> *)net_out_fwd;
		const float dt = unwarp_dt(coords_fwd.ptr->dt, NERF_CASCADES, MIN_CONE_STEPSIZE);
		float density = network_to_density(float(local_network_output[3]), density_activation);
		const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		depth_ray += weight * coords_fwd.ptr->pos.p.z();
		T *= (1.f - alpha);
		net_out_fwd += padded_output_width;
		coords_fwd += 1;
	}

	// Second pass: compute gradients
	T = 1.f;
	float depth_accumulated = 0.0f;
	for (uint32_t s = 0; s < numsteps; ++s)
	{
		const vector_t<TYPE, 4> local_network_output = *(vector_t<TYPE, 4> *)network_output;
		const float z = coords_in.ptr->pos.p.z();
		const float dt = unwarp_dt(coords_in.ptr->dt, NERF_CASCADES, MIN_CONE_STEPSIZE);
		float density = network_to_density(float(local_network_output[3]), density_activation);
		const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		depth_accumulated += weight * z;

		// suffix = sum of weighted depths after current sample (inclusive)
		const float suffix = depth_ray - depth_accumulated;

		// Gradient of depth_ray w.r.t. raw density output of sample k:
		// d(depth_ray)/d(density_k) = dt * (T_k * z_k - suffix / (1 - alpha_k) * alpha_k)
		// Simplified: dt * (T_k * z_k - suffix)
		// because suffix = sum_{j>k} w_j * z_j and d(T_j)/d(alpha_k) = -T_j/(1-alpha_k) for j>k
		float density_derivative = network_to_density_derivative(float(local_network_output[3]), density_activation);
		float dloss_by_dmlp = density_derivative * dt * (T * z - suffix) * dloss_by_ddepth;

		vector_t<TYPE, 4> local_dL_doutput;
		local_dL_doutput[0] = (TYPE)0;
		local_dL_doutput[1] = (TYPE)0;
		local_dL_doutput[2] = (TYPE)0;
		local_dL_doutput[3] = loss_scale * dloss_by_dmlp + (float(local_network_output[3]) < 0 ? -output_l1_reg_density : 0.0f);
		*(vector_t<TYPE, 4> *)dloss_doutput = local_dL_doutput;

		T *= (1.f - alpha);
		network_output += padded_output_width;
		dloss_doutput += padded_output_width;
		coords_in += 1;
	}
}
