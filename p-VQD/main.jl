using Yao, Printf, JSON

include("pvqd_functions.jl")

# INITIAL SPECIFICATIONS

N = 7 # Number of spins
depth = 5 # Depth of the ansatz
dt = 0.05 # Time step (as in the paper)
J = 0.25
B = 1

# T = 2 # Total time
# steps = round(Int, T / dt)
steps = 40

n_params = depth * (2*N - 1) + N # Number of parameters accoring to ansatz

initial_params = zeros(Float64, steps + 1, n_params)
initial_dparams = 0.01*ones(Float64, n_params)

# P-VQD EVOLUTION

max_iter = 200
learning_rate = 1
cost_ths = 5e-6
alternating = true
local_cost = true

params, cost = pvqd_train(
                        N,
                        dt,
                        steps,
                        J,
                        B,
                        alternating,
                        initial_params,
                        initial_dparams,
                        learning_rate,
                        cost_ths,
                        max_iter,
                        local_cost
                        )

S_x_op, S_y_op, S_z_op = S_xyz(N)

S_x = measure_observable(N, S_x_op, alternating, params, steps)
S_y = measure_observable(N, S_y_op, alternating, params, steps)
S_z = measure_observable(N, S_z_op, alternating, params, steps)

# EXACT EVOLUTION

H = create_hamiltonian(N, J, B)

S_x_exact = measure_observables_exact(H, dt, S_x_op, steps)
S_y_exact = measure_observables_exact(H, dt, S_y_op, steps)
S_z_exact = measure_observables_exact(H, dt, S_z_op, steps)

infidelities = calculate_infidelities(N, params, dt, steps)
# Save data

save_data = true

if save_data

	data = Dict()
    data["N"] = N
    data["depth"] = depth
    data["dt"] = dt
    data["ths"] = cost_ths
    data["steps"] = steps
    data["params"] = collect(params[:])
    data["cost"] = cost
    data["S_x"] = S_x
    data["S_y"] = S_y
    data["S_z"] = S_z
    data["S_x_exact"] = S_x_exact
    data["S_y_exact"] = S_y_exact
    data["S_z_exact"] = S_z_exact
    data["infidelities"] = infidelities

	j_data = JSON.json(data)

    filename = "data/pvqd_"*string(N)*"_spins_"*string(depth)*"_depth_"*string(dt)*"_dt_"*string(steps)*"_steps.dat"
	open(filename,"w") do f
		write(f, j_data)
	end
end

